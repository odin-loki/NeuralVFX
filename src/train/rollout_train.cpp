// Training rollout effects (include/neuralfx/rollout_train.hpp). The forward pass mirrors coarse_step() in
// src/core/rollout.cpp operation by operation; tests/test_rollout.cpp checks it against that reference and the
// gradients against finite differences.
#include <neuralfx/metrics.hpp>
#include <neuralfx/rollout_train.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <thread>

// AVX2 + FMA for the loops below, switched on after the standard headers (the trainer needs x86-64-v3; train() in
// train.cpp documents the same requirement and nvfx_experiment checks it before training).
#if defined(__GNUC__) && !defined(__clang__) && defined(__x86_64__)
#pragma GCC push_options
#pragma GCC target("arch=x86-64-v3")
#define NFX_ROLLOUT_PUSHED 1
#endif

namespace nfx::rollout {

namespace {

using Vec = std::vector<float>;
std::size_t sz(int v) { return static_cast<std::size_t>(v); }
float fl(int v) { return static_cast<float>(v); }

typedef float v8 __attribute__((vector_size(32)));
typedef float v8u __attribute__((vector_size(32), aligned(4)));

inline float dot(const float* __restrict a, const float* __restrict b, int n) {
  v8 acc{};
  int i = 0;
  for (; i + 8 <= n; i += 8) acc += *reinterpret_cast<const v8u*>(a + i) * *reinterpret_cast<const v8u*>(b + i);
  float s = 0;
  for (int k = 0; k < 8; ++k) s += acc[k];
  for (; i < n; ++i) s += a[i] * b[i];
  return s;
}

// --- convolution and physics operations on [cell][channel] layouts, with their transposes ---------------------------

void conv3(int R, const float* in, int ci, const float* W, const float* b, int co, float* out) {
  for (int y = 0; y < R; ++y) {
    for (int x = 0; x < R; ++x) {
      float* __restrict o = out + (sz(y) * sz(R) + sz(x)) * sz(co);
      for (int k = 0; k < co; ++k) o[k] = b[k];
      for (int dy = -1; dy <= 1; ++dy) {
        const int yy = y + dy;
        if (yy < 0 || yy >= R) continue;
        for (int dx = -1; dx <= 1; ++dx) {
          const int xx = x + dx;
          if (xx < 0 || xx >= R) continue;
          const float* a = in + (sz(yy) * sz(R) + sz(xx)) * sz(ci);
          const float* w = W + sz((dy + 1) * 3 + (dx + 1)) * sz(ci) * sz(co);
          for (int c = 0; c < ci; ++c) {
            const float av = a[c];
            const float* __restrict wr = w + sz(c) * sz(co);
            for (int k = 0; k < co; ++k) o[k] += wr[k] * av;
          }
        }
      }
    }
  }
}

void conv3_back(int R, const float* in, int ci, const float* W, int co, const float* g, float* gin, float* gW, float* gb) {
  for (int y = 0; y < R; ++y) {
    for (int x = 0; x < R; ++x) {
      const float* go = g + (sz(y) * sz(R) + sz(x)) * sz(co);
      for (int k = 0; k < co; ++k) gb[k] += go[k];
      for (int dy = -1; dy <= 1; ++dy) {
        const int yy = y + dy;
        if (yy < 0 || yy >= R) continue;
        for (int dx = -1; dx <= 1; ++dx) {
          const int xx = x + dx;
          if (xx < 0 || xx >= R) continue;
          const float* a = in + (sz(yy) * sz(R) + sz(xx)) * sz(ci);
          float* ga = gin ? gin + (sz(yy) * sz(R) + sz(xx)) * sz(ci) : nullptr;
          const std::size_t t = sz((dy + 1) * 3 + (dx + 1)) * sz(ci) * sz(co);
          for (int c = 0; c < ci; ++c) {
            const float av = a[c];
            float* __restrict gw = gW + t + sz(c) * sz(co);
            for (int k = 0; k < co; ++k) gw[k] += av * go[k];
            if (ga) ga[c] += dot(W + t + sz(c) * sz(co), go, co);
          }
        }
      }
    }
  }
}

// 1x1: W [in][out].
void conv1(int n, const float* in, int ci, const float* W, const float* b, int co, float* out) {
  for (int i = 0; i < n; ++i) {
    float* __restrict o = out + sz(i) * sz(co);
    for (int k = 0; k < co; ++k) o[k] = b[k];
    const float* a = in + sz(i) * sz(ci);
    for (int c = 0; c < ci; ++c) {
      const float av = a[c];
      for (int k = 0; k < co; ++k) o[k] += W[sz(c) * sz(co) + sz(k)] * av;
    }
  }
}

void conv1_back(int n, const float* in, int ci, const float* W, int co, const float* g, float* gin, float* gW, float* gb) {
  for (int i = 0; i < n; ++i) {
    const float* go = g + sz(i) * sz(co);
    const float* a = in + sz(i) * sz(ci);
    float* ga = gin + sz(i) * sz(ci);
    for (int k = 0; k < co; ++k) gb[k] += go[k];
    for (int c = 0; c < ci; ++c) {
      for (int k = 0; k < co; ++k) gW[sz(c) * sz(co) + sz(k)] += a[c] * go[k];
      ga[c] += dot(W + sz(c) * sz(co), go, co);
    }
  }
}

int cl(int v, int R) { return v < 0 ? 0 : (v >= R ? R - 1 : v); }

void divergence(int R, const float* S, int C, const float* d, int O, float qs, float* div) {
  for (int y = 0; y < R; ++y) {
    for (int x = 0; x < R; ++x) {
      const auto U = [&](int xx, int yy, int c) { return S[(sz(cl(yy, R)) * sz(R) + sz(cl(xx, R))) * sz(C) + sz(c)]; };
      div[sz(y) * sz(R) + sz(x)] = -0.5f * (U(x + 1, y, 0) - U(x - 1, y, 0) + U(x, y + 1, 1) - U(x, y - 1, 1)) + qs * d[(sz(y) * sz(R) + sz(x)) * sz(O) + sz(O - 1)];
    }
  }
}
void divergence_back(int R, int C, int O, float qs, const float* gdiv, float* gS, float* gd) {
  for (int y = 0; y < R; ++y) {
    for (int x = 0; x < R; ++x) {
      const float g = gdiv[sz(y) * sz(R) + sz(x)];
      gd[(sz(y) * sz(R) + sz(x)) * sz(O) + sz(O - 1)] += qs * g;
      gS[(sz(y) * sz(R) + sz(cl(x + 1, R))) * sz(C)] -= 0.5f * g;
      gS[(sz(y) * sz(R) + sz(cl(x - 1, R))) * sz(C)] += 0.5f * g;
      gS[(sz(cl(y + 1, R)) * sz(R) + sz(x)) * sz(C) + 1] -= 0.5f * g;
      gS[(sz(cl(y - 1, R)) * sz(R) + sz(x)) * sz(C) + 1] += 0.5f * g;
    }
  }
}

void neighbour_sum(int R, const float* p, float* o) {
  for (int y = 0; y < R; ++y) {
    for (int x = 0; x < R; ++x) {
      float s = 0;
      if (x > 0) s += p[sz(y) * sz(R) + sz(x - 1)];
      if (x < R - 1) s += p[sz(y) * sz(R) + sz(x + 1)];
      if (y > 0) s += p[sz(y - 1) * sz(R) + sz(x)];
      if (y < R - 1) s += p[sz(y + 1) * sz(R) + sz(x)];
      o[sz(y) * sz(R) + sz(x)] = s;
    }
  }
}
void jacobi(int R, int K, const float* div, float* p, float* tmp) {  // p: warm start in, solution out
  for (int k = 0; k < K; ++k) {
    neighbour_sum(R, p, tmp);
    for (int i = 0; i < R * R; ++i) p[i] = 0.25f * (div[i] + tmp[i]);
  }
}
// p_K = M^K p_0 + 0.25 sum_j M^j div with M = neighbour_sum / 4 (symmetric): the transpose runs the same iteration.
void jacobi_back(int R, int K, const float* g, float* gdiv, float* gp0, float* w, float* tmp) {
  std::memcpy(w, g, sizeof(float) * sz(R) * sz(R));
  std::fill_n(gdiv, sz(R) * sz(R), 0.f);
  for (int k = 0; k < K; ++k) {
    for (int i = 0; i < R * R; ++i) gdiv[i] += 0.25f * w[i];
    neighbour_sum(R, w, tmp);
    for (int i = 0; i < R * R; ++i) w[i] = 0.25f * tmp[i];
  }
  for (int i = 0; i < R * R; ++i) gp0[i] += w[i];
}

void subtract_grad(int R, const float* p, float* S, int C) {
  for (int y = 0; y < R; ++y) {
    for (int x = 0; x < R; ++x) {
      const auto P = [&](int xx, int yy) { return (xx < 0 || yy < 0 || xx >= R || yy >= R) ? 0.f : p[sz(yy) * sz(R) + sz(xx)]; };
      S[(sz(y) * sz(R) + sz(x)) * sz(C)] -= 0.5f * (P(x + 1, y) - P(x - 1, y));
      S[(sz(y) * sz(R) + sz(x)) * sz(C) + 1] -= 0.5f * (P(x, y + 1) - P(x, y - 1));
    }
  }
}
void subtract_grad_back(int R, const float* gS, int C, float* gp) {
  for (int y = 0; y < R; ++y) {
    for (int x = 0; x < R; ++x) {
      const float gu = gS[(sz(y) * sz(R) + sz(x)) * sz(C)], gv = gS[(sz(y) * sz(R) + sz(x)) * sz(C) + 1];
      if (x + 1 < R) gp[sz(y) * sz(R) + sz(x + 1)] -= 0.5f * gu;
      if (x > 0) gp[sz(y) * sz(R) + sz(x - 1)] += 0.5f * gu;
      if (y + 1 < R) gp[sz(y + 1) * sz(R) + sz(x)] -= 0.5f * gv;
      if (y > 0) gp[sz(y - 1) * sz(R) + sz(x)] += 0.5f * gv;
    }
  }
}

void advect(int R, const float* S, int C, float* out) {
  for (int y = 0; y < R; ++y) {
    for (int x = 0; x < R; ++x) {
      const float* s = S + (sz(y) * sz(R) + sz(x)) * sz(C);
      const float px = std::clamp(fl(x) - s[0], 0.f, fl(R - 1)), py = std::clamp(fl(y) - s[1], 0.f, fl(R - 1));
      const int x0 = std::min(static_cast<int>(px), R - 2), y0 = std::min(static_cast<int>(py), R - 2);
      const float fx = px - fl(x0), fy = py - fl(y0);
      const float* a = S + (sz(y0) * sz(R) + sz(x0)) * sz(C);
      const float* b = a + C;
      const float* c = a + sz(R) * sz(C);
      const float* d = c + C;
      float* o = out + (sz(y) * sz(R) + sz(x)) * sz(C);
      for (int k = 0; k < C; ++k) o[k] = (1.f - fy) * ((1.f - fx) * a[k] + fx * b[k]) + fy * ((1.f - fx) * c[k] + fx * d[k]);
    }
  }
}
void advect_back(int R, const float* S, int C, const float* g, float* gS) {
  for (int y = 0; y < R; ++y) {
    for (int x = 0; x < R; ++x) {
      const float* s = S + (sz(y) * sz(R) + sz(x)) * sz(C);
      const float rx = fl(x) - s[0], ry = fl(y) - s[1];
      const float px = std::clamp(rx, 0.f, fl(R - 1)), py = std::clamp(ry, 0.f, fl(R - 1));
      const int x0 = std::min(static_cast<int>(px), R - 2), y0 = std::min(static_cast<int>(py), R - 2);
      const float fx = px - fl(x0), fy = py - fl(y0);
      const std::size_t ia = (sz(y0) * sz(R) + sz(x0)) * sz(C), ib = ia + sz(C), ic = ia + sz(R) * sz(C), id = ic + sz(C);
      const float* go = g + (sz(y) * sz(R) + sz(x)) * sz(C);
      float dpx = 0, dpy = 0;
      for (int k = 0; k < C; ++k) {
        const float gk = go[k];
        gS[ia + sz(k)] += gk * (1.f - fx) * (1.f - fy);
        gS[ib + sz(k)] += gk * fx * (1.f - fy);
        gS[ic + sz(k)] += gk * (1.f - fx) * fy;
        gS[id + sz(k)] += gk * fx * fy;
        dpx += gk * ((1.f - fy) * (S[ib + sz(k)] - S[ia + sz(k)]) + fy * (S[id + sz(k)] - S[ic + sz(k)]));
        dpy += gk * ((1.f - fx) * (S[ic + sz(k)] - S[ia + sz(k)]) + fx * (S[id + sz(k)] - S[ib + sz(k)]));
      }
      if (rx > 0.f && rx < fl(R - 1)) gS[(sz(y) * sz(R) + sz(x)) * sz(C)] -= dpx;
      if (ry > 0.f && ry < fl(R - 1)) gS[(sz(y) * sz(R) + sz(x)) * sz(C) + 1] -= dpy;
    }
  }
}

// --- one differentiable coarse step ------------------------------------------------------------------------------------

struct Cache {
  Vec X, y1, a1, y2, a2, out, mid, div, p, proj, next;
  Vec g1, e1, g2, e2;
  std::vector<std::uint8_t> clamped;
};

void film(const Model& m, const StepLayout& L, std::span<const float> cond, std::size_t G, std::size_t E, Vec& g, Vec& e) {
  const int H = m.h.hidden;
  g.assign(sz(H), 1.f);
  e.assign(sz(H), 0.f);
  for (int k = 0; k < m.h.cond(); ++k) {
    for (int j = 0; j < H; ++j) {
      g[sz(j)] += m.step_w[G + sz(k) * sz(H) + sz(j)] * cond[sz(k)];
      e[sz(j)] += m.step_w[E + sz(k) * sz(H) + sz(j)] * cond[sz(k)];
    }
  }
  (void)L;
}

// in -> c.next; pressure: warm start in, solution out (also kept in c.p).
void forward(const Model& m, const StepLayout& L, const float* in, const float* noise, std::span<const float> cond, Vec& pressure, Cache& c) {
  const Hyper& h = m.h;
  const int R = h.res, N = R * R, C = h.channels(), I = h.inputs(), H = h.hidden, O = h.outputs();
  const float* w = m.step_w.data();
  c.X.resize(sz(N) * sz(I));
  for (int y = 0; y < R; ++y) {
    for (int x = 0; x < R; ++x) {
      const std::size_t i = sz(y) * sz(R) + sz(x);
      float* xi = c.X.data() + i * sz(I);
      for (int k = 0; k < C; ++k) xi[k] = in[i * sz(C) + sz(k)] / (k < kPhys ? m.scale[sz(k)] : 1.f);
      xi[C] = noise[i * kNoise];
      xi[C + 1] = noise[i * kNoise + 1];
      xi[C + 2] = (fl(x) + 0.5f) / fl(R) * 2.f - 1.f;
      xi[C + 3] = (fl(y) + 0.5f) / fl(R) * 2.f - 1.f;
    }
  }
  film(m, L, cond, L.g1, L.e1, c.g1, c.e1);
  film(m, L, cond, L.g2, L.e2, c.g2, c.e2);
  c.y1.resize(sz(N) * sz(H));
  c.a1.resize(sz(N) * sz(H));
  conv3(R, c.X.data(), I, w + L.w1, w + L.b1, H, c.y1.data());
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < H; ++j) c.a1[sz(i) * sz(H) + sz(j)] = std::max(0.f, c.g1[sz(j)] * c.y1[sz(i) * sz(H) + sz(j)] + c.e1[sz(j)]);
  }
  c.y2.resize(sz(N) * sz(H));
  c.a2.resize(sz(N) * sz(H));
  conv3(R, c.a1.data(), H, w + L.w2, w + L.b2, H, c.y2.data());
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < H; ++j) c.a2[sz(i) * sz(H) + sz(j)] = std::max(0.f, c.g2[sz(j)] * c.y2[sz(i) * sz(H) + sz(j)] + c.e2[sz(j)]);
  }
  c.out.resize(sz(N) * sz(O));
  conv1(N, c.a2.data(), H, w + L.wo, w + L.bo, O, c.out.data());
  c.mid.resize(sz(N) * sz(C));
  for (int i = 0; i < N; ++i) {
    for (int k = 0; k < kPhys; ++k) c.mid[sz(i) * sz(C) + sz(k)] = in[sz(i) * sz(C) + sz(k)] + m.scale[sz(k)] * c.out[sz(i) * sz(O) + sz(k)];
    for (int k = kPhys; k < C; ++k) c.mid[sz(i) * sz(C) + sz(k)] = std::tanh(in[sz(i) * sz(C) + sz(k)] + c.out[sz(i) * sz(O) + sz(k)]);
  }
  c.div.resize(sz(N));
  divergence(R, c.mid.data(), C, c.out.data(), O, m.qscale, c.div.data());
  Vec tmp(sz(N));
  jacobi(R, h.jacobi, c.div.data(), pressure.data(), tmp.data());
  c.p = pressure;
  c.proj = c.mid;
  subtract_grad(R, c.p.data(), c.proj.data(), C);
  c.next.resize(sz(N) * sz(C));
  advect(R, c.proj.data(), C, c.next.data());
  c.clamped.assign(sz(N) * kPhys, 0);
  for (int i = 0; i < N; ++i) {
    for (int k = 0; k < kPhys; ++k) {
      float& v = c.next[sz(i) * sz(C) + sz(k)];
      if (v < m.lo[sz(k)] || v > m.hi[sz(k)]) {
        v = std::clamp(v, m.lo[sz(k)], m.hi[sz(k)]);
        c.clamped[sz(i) * kPhys + sz(k)] = 1;
      }
    }
  }
}

// gS (gradient wrt c.next, consumed), gp_out (wrt c.p) -> gS_in, gp_in (wrt the warm start); grad accumulated.
void backward(const Model& m, const StepLayout& L, const Cache& c, std::span<const float> cond, Vec& gS, const Vec& gp_out, Vec& gS_in,
              Vec& gp_in, float* grad) {
  const Hyper& h = m.h;
  const int R = h.res, N = R * R, C = h.channels(), I = h.inputs(), H = h.hidden, O = h.outputs();
  const float* w = m.step_w.data();
  for (int i = 0; i < N; ++i) {
    for (int k = 0; k < kPhys; ++k) {
      if (c.clamped[sz(i) * kPhys + sz(k)]) gS[sz(i) * sz(C) + sz(k)] = 0;
    }
  }
  Vec gproj(sz(N) * sz(C), 0.f);
  advect_back(R, c.proj.data(), C, gS.data(), gproj.data());
  Vec gp(gp_out);
  subtract_grad_back(R, gproj.data(), C, gp.data());
  Vec gdiv(sz(N)), wk(sz(N)), tmp(sz(N));
  gp_in.assign(sz(N), 0.f);
  jacobi_back(R, h.jacobi, gp.data(), gdiv.data(), gp_in.data(), wk.data(), tmp.data());
  Vec& gmid = gproj;
  Vec gout(sz(N) * sz(O), 0.f);
  divergence_back(R, C, O, m.qscale, gdiv.data(), gmid.data(), gout.data());
  for (int i = 0; i < N; ++i) {
    for (int k = kPhys; k < C; ++k) {
      const float t = c.mid[sz(i) * sz(C) + sz(k)];
      gmid[sz(i) * sz(C) + sz(k)] *= 1.f - t * t;
    }
    for (int k = 0; k < C; ++k) gout[sz(i) * sz(O) + sz(k)] += (k < kPhys ? m.scale[sz(k)] : 1.f) * gmid[sz(i) * sz(C) + sz(k)];
  }
  gS_in = gmid;
  Vec ga2(sz(N) * sz(H), 0.f);
  conv1_back(N, c.a2.data(), H, w + L.wo, O, gout.data(), ga2.data(), grad + L.wo, grad + L.bo);
  const auto film_back = [&](const Vec& a, const Vec& y, const Vec& gam, Vec& ga, std::size_t G, std::size_t E) {
    std::vector<double> sg(sz(H), 0), se(sz(H), 0);
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < H; ++j) {
        const std::size_t q = sz(i) * sz(H) + sz(j);
        const float gz = a[q] > 0.f ? ga[q] : 0.f;
        sg[sz(j)] += static_cast<double>(gz * y[q]);
        se[sz(j)] += static_cast<double>(gz);
        ga[q] = gz * gam[sz(j)];
      }
    }
    for (int k = 0; k < h.cond(); ++k) {
      for (int j = 0; j < H; ++j) {
        grad[G + sz(k) * sz(H) + sz(j)] += static_cast<float>(sg[sz(j)]) * cond[sz(k)];
        grad[E + sz(k) * sz(H) + sz(j)] += static_cast<float>(se[sz(j)]) * cond[sz(k)];
      }
    }
  };
  film_back(c.a2, c.y2, c.g2, ga2, L.g2, L.e2);  // ga2 now holds the gradient wrt y2
  Vec ga1(sz(N) * sz(H), 0.f);
  conv3_back(R, c.a1.data(), H, w + L.w2, H, ga2.data(), ga1.data(), grad + L.w2, grad + L.b2);
  film_back(c.a1, c.y1, c.g1, ga1, L.g1, L.e1);
  Vec gX(sz(N) * sz(I), 0.f);
  conv3_back(R, c.X.data(), I, w + L.w1, H, ga1.data(), gX.data(), grad + L.w1, grad + L.b1);
  for (int i = 0; i < N; ++i) {
    for (int k = 0; k < C; ++k) gS_in[sz(i) * sz(C) + sz(k)] += gX[sz(i) * sz(I) + sz(k)] / (k < kPhys ? m.scale[sz(k)] : 1.f);
  }
}

float run_time(const Run& r, int state) { return r.t0 + static_cast<float>(state + 1) / r.p.fps; }

void state_of(const Model& m, const Run& r, int i, Vec& S) {
  const int N = m.h.res * m.h.res, C = m.h.channels();
  S.assign(sz(N) * sz(C), 0.f);
  for (int j = 0; j < N; ++j) {
    for (int k = 0; k < kPhys; ++k) S[sz(j) * sz(C) + sz(k)] = r.coarse[(sz(i) * sz(N) + sz(j)) * kPhys + sz(k)];
  }
}

}  // namespace

// --- data ---------------------------------------------------------------------------------------------------------------

void coarse_from_sim(const sim::State& st, int res, float fps, std::span<float> out) {
  const int n = st.n, k = n / res;
  if (k * res != n) throw std::invalid_argument("rollout: the coarse grid must divide the simulation grid");
  std::fill(out.begin(), out.begin() + static_cast<std::ptrdiff_t>(sz(res) * sz(res) * kPhys), 0.f);
  for (int y = 0; y < n; ++y) {
    for (int x = 0; x < n; ++x) {
      const std::size_t i = sz(y) * sz(n) + sz(x);
      float* o = out.data() + (sz(y / k) * sz(res) + sz(x / k)) * kPhys;
      o[0] += st.u[i];
      o[1] += st.v[i];
      o[2] += st.temp[i];
      o[3] += st.soot[i];
    }
  }
  const float a = 1.f / fl(k * k), av = a / (fl(k) * fps);
  for (int i = 0; i < res * res; ++i) {
    out[sz(i) * kPhys] *= av;
    out[sz(i) * kPhys + 1] *= av;
    out[sz(i) * kPhys + 2] *= a;
    out[sz(i) * kPhys + 3] *= a;
  }
}

Run record_run(const sim::Params& p, int frames, int res) {
  Run r;
  r.p = p;
  r.frames = frames;
  const std::size_t per = sz(res) * sz(res) * kPhys;
  r.coarse.resize(sz(frames) * per);
  sim::Fluid f(p);
  for (int i = 0; i < frames; ++i) {
    f.step_frame();
    coarse_from_sim(f.state(), res, p.fps, std::span(r.coarse).subspan(sz(i) * per, per));
  }
  return r;
}

// --- couplings: outside operations on the state (docs/COMPOSE.md §9) --------------------------------------------------

void apply_forcing(std::span<float> coarse, int channels, std::span<const float> f) {
  const std::size_t n = f.size() / kForce, C = sz(channels);
  for (std::size_t i = 0; i < n; ++i) {
    float* c = coarse.data() + i * C;
    const float* g = f.data() + i * kForce;
    c[2] = std::max(0.f, c[2] * g[5] + g[6]);
    c[3] = std::max(0.f, c[3] * g[5] + g[7]);
    c[1] *= g[4];
    c[0] += g[2] + g[0];
    c[1] += g[3] + g[1];
  }
}

void remove_push(std::span<float> coarse, int channels, std::span<const float> f) {
  const std::size_t n = f.size() / kForce, C = sz(channels);
  for (std::size_t i = 0; i < n; ++i) {
    coarse[i * C] -= f[i * kForce];
    coarse[i * C + 1] -= f[i * kForce + 1];
  }
}

namespace {

float smooth01(float t) {
  t = std::clamp(t, 0.f, 1.f);
  return t * t * (3.f - 2.f * t);
}

constexpr float kCoarseCells = 32.f;  // push and force amplitudes are in cells of a 32-cell grid per frame

}  // namespace

float Coupling::envelope(int state) const {
  if (state < onset || state >= onset + duration) return 0.f;
  const float ramp = std::clamp(fl(duration) / 3.f, 1.f, 4.f);
  return smooth01(fl(std::min(state - onset + 1, onset + duration - state)) / ramp);
}

bool ForcingSpec::active(int state) const {
  return std::ranges::any_of(events, [state](const Coupling& c) { return c.envelope(state) > 0.f; });
}

void forcing_fields(const ForcingSpec& spec, const sim::Params& p, int n, int state, SimForcing& out) {
  const std::size_t nn = sz(n) * sz(n);
  out.n = n;
  for (auto* v : {&out.pu, &out.pv, &out.fu, &out.fv, &out.ah, &out.as}) v->assign(nn, 0.f);
  out.vk.assign(nn, 1.f);
  out.mk.assign(nn, 1.f);
  out.push = out.force = out.material = false;
  const float to_sim = fl(n) / kCoarseCells * p.fps;  // cells of the 32-cell grid per frame -> solver cells per second
  for (const Coupling& c : spec.events) {
    const float env = c.envelope(state);
    if (env <= 0.f) continue;
    const float age = fl(state - c.onset);
    const float cx = c.x + c.dx * age, cy = c.y + c.dy * age, r = std::max(1e-3f, c.radius);
    const float ca = std::cos(c.angle), sa = std::sin(c.angle);
    for (int y = 0; y < n; ++y) {
      for (int x = 0; x < n; ++x) {
        const std::size_t i = sz(y) * sz(n) + sz(x);
        const float X = (fl(x) + 0.5f) / fl(n), Y = (fl(y) + 0.5f) / fl(n);
        const float ex = X - cx, ey = Y - cy, d2 = ex * ex + ey * ey;
        const float gauss = std::exp(-0.5f * d2 / (r * r));
        switch (c.kind) {
          case Coupling::Kind::push:
          case Coupling::Kind::force: {
            float u = 0.f, v = 0.f;
            if (c.shape == Coupling::Shape::gust) {
              u = c.amp * gauss * ca;
              v = c.amp * gauss * sa;
            } else if (c.shape == Coupling::Shape::vortex) {  // peak speed amp at the radius, as compose's vortex field
              const float d = std::sqrt(d2);
              const float sp = c.amp * (d / r) * std::exp(0.5f * (1.f - d2 / (r * r))) / (d + 1e-4f);
              u = -sp * ey;
              v = sp * ex;
            } else {  // a wave: a flow along `angle`, modulated across the domain and drifting (wind, shear, gusts)
              const float ph = 6.2831853f * (((X - c.x) * std::cos(c.across) + (Y - c.y) * std::sin(c.across)) / std::max(1e-3f, c.wavelength) - c.speed * age);
              const float k = 0.5f + 0.5f * std::sin(ph);
              u = c.amp * k * ca;
              v = c.amp * k * sa;
            }
            auto& U = c.kind == Coupling::Kind::push ? out.pu : out.fu;
            auto& V = c.kind == Coupling::Kind::push ? out.pv : out.fv;
            U[i] += env * to_sim * u;
            V[i] += env * to_sim * v;
            (c.kind == Coupling::Kind::push ? out.push : out.force) = true;
            break;
          }
          case Coupling::Kind::ceiling:
            out.vk[i] *= 1.f - env * c.amp * smooth01((Y - c.height) / std::max(1e-3f, c.soft));
            out.force = true;
            break;
          case Coupling::Kind::add:
            out.ah[i] += env * c.amp * gauss;
            out.as[i] += env * c.amp2 * gauss;
            out.material = true;
            break;
          case Coupling::Kind::remove:
            out.mk[i] *= 1.f - env * c.amp * (c.band ? smooth01((Y - c.height) / std::max(1e-3f, c.soft)) : gauss);
            out.material = true;
            break;
        }
      }
    }
  }
}

void apply_forcing(sim::Fluid& f, const SimForcing& s) {
  const std::size_t nn = sz(s.n) * sz(s.n);
  if (s.material) {
    const sim::State st = f.state();
    std::vector<float> dt(nn), ds(nn);
    for (std::size_t i = 0; i < nn; ++i) {
      dt[i] = st.temp[i] * (s.mk[i] - 1.f) + s.ah[i];
      ds[i] = st.soot[i] * (s.mk[i] - 1.f) + s.as[i];
    }
    f.add_material(dt, ds);
  }
  if (s.push || s.force) {
    const sim::State st = f.state();
    std::vector<float> du(nn), dv(nn);
    for (std::size_t i = 0; i < nn; ++i) {
      du[i] = s.fu[i] + s.pu[i];
      dv[i] = st.v[i] * (s.vk[i] - 1.f) + s.fv[i] + s.pv[i];
    }
    f.push(du, dv);
  }
}

void unpush(sim::Fluid& f, const SimForcing& s) {
  if (!s.push) return;
  const std::size_t nn = sz(s.n) * sz(s.n);
  std::vector<float> du(nn), dv(nn);
  for (std::size_t i = 0; i < nn; ++i) {
    du[i] = -s.pu[i];
    dv[i] = -s.pv[i];
  }
  f.push(du, dv);
}

void coarse_forcing(const SimForcing& s, int res, float fps, std::span<float> out) {
  const int n = s.n, k = n / res;
  if (k * res != n) throw std::invalid_argument("rollout: the coarse grid must divide the simulation grid");
  std::fill(out.begin(), out.begin() + static_cast<std::ptrdiff_t>(sz(res) * sz(res) * kForce), 0.f);
  for (int y = 0; y < n; ++y) {
    for (int x = 0; x < n; ++x) {
      const std::size_t i = sz(y) * sz(n) + sz(x);
      float* o = out.data() + (sz(y / k) * sz(res) + sz(x / k)) * kForce;
      o[0] += s.pu[i];
      o[1] += s.pv[i];
      o[2] += s.fu[i];
      o[3] += s.fv[i];
      o[4] += s.vk[i];
      o[5] += s.mk[i];
      o[6] += s.ah[i];
      o[7] += s.as[i];
    }
  }
  const float a = 1.f / fl(k * k), av = a / (fl(k) * fps);  // as coarse_from_sim
  for (int i = 0; i < res * res; ++i) {
    float* o = out.data() + sz(i) * kForce;
    for (int c = 0; c < 4; ++c) o[c] *= av;
    for (int c = 4; c < kForce; ++c) o[c] *= a;
  }
}

Run record_forced_run(const sim::Params& p, const ForcingSpec& spec, int frames, int res) {
  Run r;
  r.p = p;
  r.frames = frames;
  const std::size_t per = sz(res) * sz(res) * kPhys, fper = sz(res) * sz(res) * kForce;
  r.coarse.resize(sz(frames) * per);
  r.forcing_at.assign(sz(frames), -1);
  sim::Fluid f(p);
  SimForcing sf;
  const int n = p.sim_res > 0 ? p.sim_res : p.size;
  for (int i = 0; i < frames; ++i) {
    const bool on = spec.active(i);
    if (on) {
      forcing_fields(spec, p, n, i, sf);
      apply_forcing(f, sf);
      r.forcing_at[sz(i)] = static_cast<int>(r.forcing.size() / fper);
      r.forcing.resize(r.forcing.size() + fper);
      coarse_forcing(sf, res, p.fps, std::span(r.forcing).subspan(r.forcing.size() - fper, fper));
    }
    f.step_frame();
    if (on) unpush(f, sf);
    coarse_from_sim(f.state(), res, p.fps, std::span(r.coarse).subspan(sz(i) * per, per));
  }
  return r;
}

Run record_handover_run(const sim::Params& from, int before, const sim::Params& p, int frames, int res) {
  sim::Fluid a(from);
  for (int i = 0; i < before; ++i) a.step_frame();
  sim::Params q = p;
  q.sim_res = from.sim_res > 0 ? from.sim_res : from.size;
  sim::Fluid f(q);
  f.set_state(a.state());
  Run r;
  r.p = q;
  r.frames = frames;
  r.t0 = fl(before - 1) / p.fps;  // state 0 (the handed-over state) is at the explosion's time, before / fps
  const std::size_t per = sz(res) * sz(res) * kPhys;
  r.coarse.resize(sz(frames) * per);
  coarse_from_sim(f.state(), res, p.fps, std::span(r.coarse).subspan(0, per));
  for (int i = 1; i < frames; ++i) {
    f.step_frame();
    coarse_from_sim(f.state(), res, p.fps, std::span(r.coarse).subspan(sz(i) * per, per));
  }
  return r;
}

// What the fireball scene does to each model (nvfx_fireball, measured per frame on the coarse grid; push in cells of the
// 32-cell grid per frame): the wreck's fire is pushed by up to 0.49 (median 0.18 at its strongest cell) against its
// own flow of 0.35 to 0.5 RMS, and loses half of the material in its top four rows every frame (transfer); the
// clouds are damped by the ceiling (v multiplied by down to 0.7 per frame, changes up to 0.1), receive up to 0.06 heat
// and soot per frame (transfers), lose 5% per frame (the second cloud) or everything in a box over the smoke source
// (suppress); the second explosion is pushed by up to 0.09; the smoke model takes over explosion states (hand-over
// runs). The random specs cover these ranges and go somewhat beyond them, for every effect.
ForcingSpec random_forcing(sim::Effect e, int frames, std::uint64_t seed, int first, int last) {
  std::mt19937_64 rng(seed * 0x9E3779B97F4A7C15ULL + 0x1F0BCE5ULL);
  const auto U = [&](float a, float b) { return std::uniform_real_distribution<float>(a, b)(rng); };
  const auto I = [&](int a, int b) { return a + static_cast<int>(rng() % static_cast<std::uint64_t>(std::max(1, b - a + 1))); };
  const bool fire = e == sim::Effect::fire;
  const float push_max = fire ? 0.5f : 0.3f;
  last = std::clamp(last, first, frames - 2);
  ForcingSpec spec;
  const int n = I(1, 4);
  for (int k = 0; k < n; ++k) {
    Coupling c;
    const float w = U(0.f, 1.f);  // kind: push 35%, lasting force 15%, ceiling 15%, material in 15%, out 20%
    c.kind = w < 0.35f ? Coupling::Kind::push
             : w < 0.50f ? Coupling::Kind::force
             : w < 0.65f ? Coupling::Kind::ceiling
             : w < 0.80f ? Coupling::Kind::add
                         : Coupling::Kind::remove;
    c.onset = I(first, last);
    c.x = U(0.2f, 0.8f);
    c.y = U(0.15f, 0.85f);
    c.dx = U(-0.006f, 0.006f);
    c.dy = U(-0.006f, 0.006f);
    switch (c.kind) {
      case Coupling::Kind::push:
      case Coupling::Kind::force: {
        const float s = U(0.f, 1.f);
        c.shape = s < 0.4f ? Coupling::Shape::gust : s < 0.7f ? Coupling::Shape::vortex : Coupling::Shape::wave;
        c.angle = U(0.f, 6.2831853f);
        c.across = U(0.f, 6.2831853f);
        c.wavelength = U(0.4f, 3.f);
        c.speed = U(-0.03f, 0.03f);
        c.radius = c.shape == Coupling::Shape::vortex ? U(0.08f, 0.25f) : U(0.12f, 0.4f);
        if (c.kind == Coupling::Kind::push) {  // a flow passing through, for a while
          c.duration = I(10, 120);
          c.amp = U(0.05f, push_max);
        } else {  // a short kick that stays in the flow
          c.duration = I(2, 12);
          c.amp = U(0.01f, 0.4f * push_max) / static_cast<float>(c.duration);  // 0.01 to 0.2 (fire) in all
        }
        if (c.shape == Coupling::Shape::vortex && U(0.f, 1.f) < 0.5f) c.amp = -c.amp;
        break;
      }
      case Coupling::Kind::ceiling:
        c.duration = I(30, 150);
        c.amp = U(0.05f, 0.3f);
        c.height = U(0.3f, 0.8f);
        c.soft = U(0.1f, 0.3f);
        break;
      case Coupling::Kind::add:  // fire's soot is thin (its training range ends at 0.16): less of it
        c.amp = U(0.005f, 0.06f);
        c.amp2 = fire ? U(0.0005f, 0.004f) : U(0.005f, 0.06f);
        c.duration = I(10, std::clamp(static_cast<int>(1.f / std::max(c.amp, c.amp2)), 10, 60));  // at most about 1 in all
        c.radius = U(0.06f, 0.2f);
        break;
      case Coupling::Kind::remove:
        c.duration = I(10, 120);
        c.band = U(0.f, 1.f) < 0.4f;  // out through the top, or a disc anywhere (suppress, transfer out)
        c.amp = U(0.f, 1.f) < 0.2f ? 1.f : U(0.02f, 0.6f);
        c.height = U(0.7f, 0.9f);
        c.soft = U(0.03f, 0.12f);
        c.radius = U(0.08f, 0.25f);
        if (!c.band && U(0.f, 1.f) < 0.3f) {  // over the source
          c.x = 0.5f;
          c.y = U(0.05f, 0.2f);
          c.dx = c.dy = 0.f;
        }
        break;
    }
    c.duration = std::min(c.duration, frames - c.onset);
    spec.events.push_back(c);
  }
  return spec;
}

// Strong pushes as scenes give them (study I round 2, docs/COMPOSE.md §10). In cells of the 32-cell grid per frame:
// the wall of examples/scenes/firewall.nvfxs feels the vortex ring at up to 0.67 (4 world pixels a frame at 6 pixels a
// cell) and its sky 0.67 from the ring and 0.2 +- 70% from the gale; the fireball's wreck is pushed by up to 0.49 and
// loses half of its top rows to the sky every frame; a sustained gust of a fraction of a cell per frame weakens the
// learned fire within seconds (§5). Here a gale always blows (a wave 2 to 6 domain lengths long, so it varies slowly
// across the domain and drifts like gusts; within 20 degrees of sideways; for 60 to 150 frames), a ring rolls through
// in 60% of the runs (in 22 to 44 frames), a blast (a broad gust of 5 to 20 frames) comes in 30%, and the material
// leaves through the top in 50% (fire) or 30% (smoke and explosions).
ForcingSpec scene_forcing(sim::Effect e, int frames, std::uint64_t seed, int first, int last) {
  std::mt19937_64 rng(seed * 0xD1B54A32D192ED03ULL + 0x5CE7E5ULL);
  const auto U = [&](float a, float b) { return std::uniform_real_distribution<float>(a, b)(rng); };
  const auto I = [&](int a, int b) { return a + static_cast<int>(rng() % static_cast<std::uint64_t>(std::max(1, b - a + 1))); };
  const bool fire = e == sim::Effect::fire;
  // Rings stop at 0.7 on both: stronger ones (up to 1.0 was tried on fire) are spun up by the simulator's vorticity
  // confinement to flows of 1.5 to 3.8 cells per frame, beyond anything v1 saw and beyond its clamp.
  const float gale_max = fire ? 0.6f : 0.4f, ring_max = 0.7f;
  last = std::clamp(last, first, frames - 2);
  ForcingSpec spec;
  const auto clip = [&](Coupling& c) {
    c.duration = std::max(1, std::min(c.duration, frames - c.onset));
    spec.events.push_back(c);
  };
  const float side = U(0.f, 1.f) < 0.5f ? 0.f : 3.14159265f;  // the wind blows to the right or to the left
  {  // the gale
    Coupling c;
    c.kind = Coupling::Kind::push;
    c.shape = Coupling::Shape::wave;
    c.onset = I(first, std::min(first + 10, last));
    c.duration = I(60, 150);
    c.angle = side + U(-0.35f, 0.35f);
    c.across = U(0.f, 6.2831853f);
    c.wavelength = U(2.f, 6.f);
    c.speed = U(-0.01f, 0.01f);
    c.x = U(0.f, 1.f);
    c.y = U(0.f, 1.f);
    c.amp = U(0.15f, gale_max);
    clip(c);
  }
  if (U(0.f, 1.f) < 0.6f) {  // a vortex ring seen from the side: two opposite vortices travelling with the wind
    // It rolls through the domain at 0.03 to 0.06 of its width per frame (the firewall's ring crosses a tile of the wall
    // at 0.05), entering at one side and leaving at the other.
    const float speed = U(0.03f, 0.06f), sep = U(0.08f, 0.15f), core = U(0.05f, 0.1f), y = U(0.3f, 0.75f), amp = U(0.3f, ring_max);
    const int onset = I(first, last), duration = static_cast<int>(std::ceil(1.3f / speed));
    const float dir = side == 0.f ? 1.f : -1.f, x0 = dir > 0.f ? U(-0.15f, 0.f) : U(1.f, 1.15f);
    for (const float sgn : {1.f, -1.f}) {
      Coupling c;
      c.kind = Coupling::Kind::push;
      c.shape = Coupling::Shape::vortex;
      c.onset = onset;
      c.duration = duration;
      c.x = x0;
      c.y = y + sgn * sep;
      c.dx = dir * speed;
      c.radius = core;
      c.amp = sgn * dir * amp;  // the upper core turns so that the pair blows along the travel between the cores
      clip(c);
    }
  }
  if (U(0.f, 1.f) < 0.3f) {  // a blast: a broad gust for a few frames
    Coupling c;
    c.kind = Coupling::Kind::push;
    c.shape = Coupling::Shape::gust;
    c.onset = I(first, last);
    c.duration = I(5, 20);
    c.x = U(0.2f, 0.8f);
    c.y = U(0.2f, 0.8f);
    c.radius = U(0.3f, 0.6f);
    c.angle = U(0.f, 6.2831853f);
    c.amp = U(0.2f, fire ? 0.5f : 0.3f);
    clip(c);
  }
  if (U(0.f, 1.f) < (fire ? 0.5f : 0.3f)) {  // material out through the top (a transfer to the sky), for good
    Coupling c;
    c.kind = Coupling::Kind::remove;
    c.band = true;
    c.onset = I(first, last);
    c.duration = frames;
    c.amp = U(0.2f, 0.6f);
    c.height = U(0.75f, 0.88f);
    c.soft = U(0.03f, 0.08f);
    clip(c);
  }
  return spec;
}

// --- stepper ------------------------------------------------------------------------------------------------------------

double window_loss(const Model& m, const Run& r, int first, int unroll, int burn, float sigma, float profile, std::uint64_t noise_seed,
                   std::vector<float>* grad, float activity, const Model* anchor, float anchor_weight) {
  const Hyper& h = m.h;
  const StepLayout L = step_layout(h);
  const int R = h.res, N = R * R, C = h.channels();
  Vec S, pressure(sz(N), 0.f), noise(sz(N) * kNoise), cond(sz(h.cond()));
  state_of(m, r, first, S);
  const std::vector<float> controls = r.controls();
  // Couplings (forced runs): the operations before the step that produced state i are applied to the state before the
  // step from state i - 1, and the push is taken out after it (as compose does at run time).
  const std::size_t fper = sz(N) * kForce;
  const auto forcing = [&](int state) -> std::span<const float> {
    const float* f = r.forcing_for(state, fper);
    return f ? std::span<const float>(f, fper) : std::span<const float>{};
  };
  int i0 = first;
  if (burn > 0) {  // the stepper's own rollout, no gradient: it learns to correct its own drift
    Cache cb;
    for (int s = 0; s < burn; ++s) {
      condition(m, controls, run_time(r, i0 + s), cond);
      coarse_noise(m, r.p.seed, run_time(r, i0 + s), noise);
      const auto f = forcing(i0 + s + 1);
      if (!f.empty()) apply_forcing(S, C, f);
      forward(m, L, S.data(), noise.data(), cond, pressure, cb);
      S = cb.next;
      if (!f.empty()) remove_push(S, C, f);
    }
    i0 += burn;
  }
  std::vector<Cache> cs(sz(unroll));
  std::vector<Vec> conds(sz(unroll));
  std::vector<Vec> gout(sz(unroll));
  std::vector<Vec> gprev(sz(unroll), Vec(activity > 0.f ? sz(N) * sz(C) : 0, 0.f));  // activity terms on earlier outputs
  std::vector<std::span<const float>> fs(sz(unroll));  // the operations before each step
  std::vector<std::vector<std::uint8_t>> zeroed(sz(unroll));  // cells whose heat (bit 0) or soot (bit 1) was clamped at zero
  std::vector<Vec> outs(sz(unroll));  // each step's state after the push is taken out (what the loss sees)
  std::array<float, kPhys> wch{};
  for (int k = 0; k < kPhys; ++k) wch[sz(k)] = 1.f / (m.scale[sz(k)] * m.scale[sz(k)]);
  const float inv = 1.f / (fl(N) * kPhys * fl(unroll));
  std::mt19937 rng(static_cast<unsigned>(noise_seed));
  std::normal_distribution<float> nd(0.f, 1.f);
  double loss = 0;
  for (int s = 0; s < unroll; ++s) {
    conds[sz(s)].resize(sz(h.cond()));
    condition(m, controls, run_time(r, i0 + s), conds[sz(s)]);
    coarse_noise(m, r.p.seed, run_time(r, i0 + s), noise);
    if (sigma > 0.f) {
      for (int j = 0; j < N; ++j) {
        for (int k = 0; k < kPhys; ++k) {
          float& x = S[sz(j) * sz(C) + sz(k)];
          x = std::clamp(x + sigma * m.scale[sz(k)] * nd(rng), m.lo[sz(k)], m.hi[sz(k)]);
        }
      }
    }
    fs[sz(s)] = forcing(i0 + s + 1);
    if (!fs[sz(s)].empty()) {
      const std::span<const float> f = fs[sz(s)];
      auto& z = zeroed[sz(s)];
      z.assign(sz(N), 0);
      for (int j = 0; j < N; ++j) {
        const float* g = f.data() + sz(j) * kForce;
        const float* x = S.data() + sz(j) * sz(C);
        z[sz(j)] = static_cast<std::uint8_t>((x[2] * g[5] + g[6] < 0.f ? 1 : 0) | (x[3] * g[5] + g[7] < 0.f ? 2 : 0));
      }
      apply_forcing(S, C, f);
    }
    Vec anchored;  // the anchor model's first step from the same input (round 2: plain windows from a true state)
    if (s == 0 && burn == 0 && anchor && anchor_weight > 0.f) {
      Vec ap(pressure);
      Cache ca;
      forward(*anchor, L, S.data(), noise.data(), conds[0], ap, ca);
      anchored = std::move(ca.next);
      if (!fs[0].empty()) remove_push(anchored, C, fs[0]);
    }
    forward(m, L, S.data(), noise.data(), conds[sz(s)], pressure, cs[sz(s)]);
    S = cs[sz(s)].next;
    if (!fs[sz(s)].empty()) remove_push(S, C, fs[sz(s)]);
    if (activity > 0.f) outs[sz(s)] = S;
    const float* t = r.coarse.data() + sz(i0 + s + 1) * sz(N) * kPhys;
    Vec& g = gout[sz(s)];
    g.assign(sz(N) * sz(C), 0.f);
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < kPhys; ++k) {
        const float e = S[sz(j) * sz(C) + sz(k)] - t[sz(j) * kPhys + sz(k)];
        loss += static_cast<double>(wch[sz(k)] * e * e * inv);
        g[sz(j) * sz(C) + sz(k)] = 2.f * wch[sz(k)] * e * inv;
      }
    }
    if (!anchored.empty()) {  // anchor_weight * mean over cells and physical channels of the scaled squared difference
      const float ainv = anchor_weight / (fl(N) * kPhys);
      for (int j = 0; j < N; ++j) {
        for (int k = 0; k < kPhys; ++k) {
          const float e = S[sz(j) * sz(C) + sz(k)] - anchored[sz(j) * sz(C) + sz(k)];
          loss += static_cast<double>(wch[sz(k)] * e * e * ainv);
          g[sz(j) * sz(C) + sz(k)] += 2.f * wch[sz(k)] * e * ainv;
        }
      }
    }
    if (profile > 0.f) {  // where the heat and soot are: row and column sums
      const float pinv = profile / fl(4 * R * unroll);
      for (int k = 2; k < kPhys; ++k) {
        for (int axis = 0; axis < 2; ++axis) {
          Vec dm(sz(R), 0.f);
          for (int y = 0; y < R; ++y) {
            for (int x = 0; x < R; ++x) {
              const int j = y * R + x;
              dm[sz(axis == 0 ? y : x)] += (S[sz(j) * sz(C) + sz(k)] - t[sz(j) * kPhys + sz(k)]) / (m.scale[sz(k)] * fl(R));
            }
          }
          for (int b = 0; b < R; ++b) loss += static_cast<double>(pinv * dm[sz(b)] * dm[sz(b)]);
          for (int y = 0; y < R; ++y) {
            for (int x = 0; x < R; ++x) {
              const int j = y * R + x;
              g[sz(j) * sz(C) + sz(k)] += 2.f * pinv * dm[sz(axis == 0 ? y : x)] / (m.scale[sz(k)] * fl(R));
            }
          }
        }
      }
    }
    if (activity > 0.f && s > 0) {  // how much each channel changes per frame: mean squared change, model against truth
      const float* tp = r.coarse.data() + sz(i0 + s) * sz(N) * kPhys;  // the truth one frame earlier
      const Vec& prev = outs[sz(s - 1)];
      for (int k = 0; k < kPhys; ++k) {
        const float is2 = 1.f / (m.scale[sz(k)] * m.scale[sz(k)]);
        double am = 0, at = 0;
        for (int j = 0; j < N; ++j) {
          const float dmod = S[sz(j) * sz(C) + sz(k)] - prev[sz(j) * sz(C) + sz(k)];
          const float dtru = t[sz(j) * kPhys + sz(k)] - tp[sz(j) * kPhys + sz(k)];
          am += static_cast<double>(dmod * dmod * is2);
          at += static_cast<double>(dtru * dtru * is2);
        }
        am /= N;
        at /= N;
        const float e = static_cast<float>(am - at), w = activity / (fl(kPhys) * fl(unroll - 1));
        loss += static_cast<double>(w * e * e);
        for (int j = 0; j < N; ++j) {
          const float dmod = S[sz(j) * sz(C) + sz(k)] - prev[sz(j) * sz(C) + sz(k)];
          const float gj = w * 2.f * e * 2.f * dmod * is2 / fl(N);
          g[sz(j) * sz(C) + sz(k)] += gj;
          gprev[sz(s - 1)][sz(j) * sz(C) + sz(k)] -= gj;
        }
      }
    }
  }
  if (grad) {
    Vec gS(sz(N) * sz(C), 0.f), gp(sz(N), 0.f), gS_in, gp_in;
    for (int s = unroll - 1; s >= 0; --s) {
      for (std::size_t j = 0; j < gS.size(); ++j) gS[j] += gout[sz(s)][j];
      if (activity > 0.f) {
        for (std::size_t j = 0; j < gS.size(); ++j) gS[j] += gprev[sz(s)][j];
      }
      backward(m, L, cs[sz(s)], conds[sz(s)], gS, gp, gS_in, gp_in, grad->data());
      if (!fs[sz(s)].empty()) {  // through the operations: the multipliers scale, the offsets pass, clamped cells stop
        const std::span<const float> f = fs[sz(s)];
        for (int j = 0; j < N; ++j) {
          const float* g = f.data() + sz(j) * kForce;
          float* q = gS_in.data() + sz(j) * sz(C);
          const std::uint8_t z = zeroed[sz(s)][sz(j)];
          q[1] *= g[4];
          q[2] = z & 1 ? 0.f : q[2] * g[5];
          q[3] = z & 2 ? 0.f : q[3] * g[5];
        }
      }
      gS.swap(gS_in);
      gp.swap(gp_in);
    }
  }
  return loss;
}

bool window_has_coupling(const Run& r, int first, int unroll) {
  for (int i = first + 1; i <= first + unroll; ++i) {
    if (i >= 0 && sz(i) < r.forcing_at.size() && r.forcing_at[sz(i)] >= 0) return true;
  }
  return false;
}

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

int thread_count(int requested) {
  const int hw = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  return requested > 0 ? requested : hw;
}

}  // namespace

StepperResult train_stepper(Model& m, std::span<const Run> runs, const StepperOptions& o) {
  if (runs.empty()) throw std::invalid_argument("rollout: no runs");
  const Hyper& h = m.h;
  const std::size_t per = sz(h.res) * sz(h.res) * kPhys;
  for (const Run& r : runs) {
    if (r.coarse.size() != sz(r.frames) * per || r.frames < o.max_unroll + 2) throw std::invalid_argument("rollout: run has the wrong shape");
  }
  // Channel scales (RMS) and the range the state is kept in (1.5 x the largest magnitude seen).
  std::array<double, kPhys> s2{};
  std::array<float, kPhys> mx{};
  std::size_t count = 0;
  for (const Run& r : runs) {
    for (std::size_t j = 0; j < r.coarse.size(); j += kPhys) {
      for (int k = 0; k < kPhys; ++k) {
        const float v = r.coarse[j + sz(k)];
        s2[sz(k)] += static_cast<double>(v) * v;
        mx[sz(k)] = std::max(mx[sz(k)], std::abs(v));
      }
      ++count;
    }
  }
  for (int k = 0; k < kPhys && !o.keep_normalisation; ++k) {
    m.scale[sz(k)] = static_cast<float>(std::sqrt(s2[sz(k)] / static_cast<double>(count))) + 1e-4f;
    m.hi[sz(k)] = 1.5f * mx[sz(k)] + 1e-4f;
    m.lo[sz(k)] = k < 2 ? -m.hi[sz(k)] : 0.f;
  }
  const StepLayout L = step_layout(h);
  if (m.step_w.size() != L.size) throw std::invalid_argument("rollout: weights do not match the hyperparameters");
  const int threads = thread_count(o.threads);
  Adam adam(L.size);
  StepperResult res;
  const auto t0 = std::chrono::steady_clock::now();
  const int total = o.iterations + o.finetune + o.activity_stage;
  std::vector<double> tail;
  // The mix of plain and coupled runs (o.plain_runs >= 0): a window from the coupled runs with probability
  // o.coupled_share.
  const std::size_t n_plain = o.plain_runs < 0 ? runs.size() : std::min(runs.size(), sz(o.plain_runs));
  if (o.plain_runs >= 0 && ((n_plain == 0 && o.coupled_share < 1.f) || (n_plain == runs.size() && o.coupled_share > 0.f))) {
    throw std::invalid_argument("rollout: the mix needs plain and coupled runs");
  }
  const auto pick_run = [&](std::mt19937_64& rng) {
    const bool coupled = std::uniform_real_distribution<float>(0.f, 1.f)(rng) < o.coupled_share;
    return coupled ? n_plain + rng() % (runs.size() - n_plain) : rng() % n_plain;
  };
  for (int it = 0; it < total; ++it) {
    const bool fine = it >= o.iterations;
    const bool act = it >= o.iterations + o.finetune;
    const int unroll = fine ? o.max_unroll : std::min(o.max_unroll, 1 + it * o.max_unroll / std::max(1, o.iterations / 2));
    std::vector<Vec> grads(sz(threads), Vec(L.size, 0.f));
    std::vector<double> losses(sz(threads), 0.0);
    {
      std::vector<std::jthread> pool;
      for (int t = 0; t < threads; ++t) {
        pool.emplace_back([&, t] {
          std::mt19937_64 rng(o.seed * 1000003ULL + static_cast<std::uint64_t>(it) * 977ULL + static_cast<std::uint64_t>(t));
          for (int b = t; b < o.batch; b += threads) {
            const std::size_t ri = o.plain_runs < 0 ? rng() % runs.size() : pick_run(rng);
            const Run& r = runs[ri];
            const int burn = fine && o.burn_max > 0 && (b & 1) ? static_cast<int>(rng() % static_cast<std::uint64_t>(o.burn_max + 1)) : 0;
            const int span = r.frames - unroll - burn - 1;
            if (span <= 0) continue;
            int first = static_cast<int>(rng() % static_cast<std::uint64_t>(span));
            if (o.aim_couplings && !r.forcing_at.empty()) {  // round 2: a window in which a coupling acts
              for (int k = 0; k < 16 && !window_has_coupling(r, first + burn, unroll); ++k) first = static_cast<int>(rng() % static_cast<std::uint64_t>(span));
            }
            const bool anchored = o.anchor > 0.f && o.anchor_model && o.plain_runs >= 0 && ri < n_plain;
            losses[sz(t)] += window_loss(m, r, first, unroll, burn, o.sigma, fine ? o.profile : 0.f, rng(), &grads[sz(t)], act ? o.activity : 0.f,
                                         anchored ? o.anchor_model : nullptr, anchored ? o.anchor : 0.f);
          }
        });
      }
    }
    for (int t = 1; t < threads; ++t) {
      for (std::size_t j = 0; j < L.size; ++j) grads[0][j] += grads[sz(t)][j];
    }
    double loss = 0, gn = 0;
    for (const double v : losses) loss += v;
    loss /= o.batch;
    for (float& g : grads[0]) {
      g /= static_cast<float>(o.batch);
      gn += static_cast<double>(g) * g;
    }
    gn = std::sqrt(gn);
    if (gn > o.clip) {
      const float s = static_cast<float>(o.clip / gn);
      for (float& g : grads[0]) g *= s;
    }
    const int stage_it = act ? it - o.iterations - o.finetune : (fine ? it - o.iterations : it);
    const int stage_n = act ? o.activity_stage : (fine ? o.finetune : o.iterations);
    const float base = fine ? o.lr_finetune : o.lr;
    const float lr = base * (0.05f + 0.95f * 0.5f * (1.f + std::cos(3.14159265f * static_cast<float>(stage_it) / static_cast<float>(std::max(1, stage_n)))));
    adam.step(m.step_w, grads[0], lr);
    if (it >= total - std::max(1, total / 20)) tail.push_back(loss);
    if (it % std::max(1, o.log_every) == 0 || it == total - 1) {
      res.curve.emplace_back(it, loss);
      if (o.progress) o.progress(it, unroll, loss);
    }
    if (o.checkpoint && o.checkpoint_every > 0 && (it + 1) % o.checkpoint_every == 0) o.checkpoint(it + 1);
    if (o.stop_after > 0 && it + 1 >= o.stop_after) break;
  }
  for (const double v : tail) res.final_loss += v / static_cast<double>(tail.size());
  res.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return res;
}

// --- renderer -----------------------------------------------------------------------------------------------------------

RenderSample render_sample(const sim::Fluid& f, int res) {
  RenderSample s;
  const sim::State st = f.state();
  s.size = f.params().size;
  if (st.n != s.size) throw std::invalid_argument("rollout: render samples need the solver at the output size");
  s.fine_t = st.temp;
  s.fine_d = st.soot;
  s.coarse.resize(sz(res) * sz(res) * kPhys);
  coarse_from_sim(st, res, f.params().fps, s.coarse);
  s.rgba.resize(sz(s.size) * sz(s.size) * 4);
  f.render(s.rgba);
  return s;
}

namespace {

State sample_state(const Model& m, const RenderSample& s) {
  State st;
  st.res = m.h.res;
  st.size = s.size;
  const int C = m.h.channels(), N = m.h.res * m.h.res;
  st.coarse.assign(sz(N) * sz(C), 0.f);
  for (int i = 0; i < N; ++i) {
    for (int k = 0; k < kPhys; ++k) st.coarse[sz(i) * sz(C) + sz(k)] = s.coarse[sz(i) * kPhys + sz(k)];
  }
  st.fine_t = s.fine_t;
  st.fine_d = s.fine_d;
  return st;
}

}  // namespace

double train_renderer(Model& m, std::span<const RenderSample> samples, const RendererOptions& o) {
  if (samples.empty()) throw std::invalid_argument("rollout: no render samples");
  // Input normalisation: the 99.5th percentile of fine heat and soot over the samples' active pixels.
  {
    std::vector<float> t, d;
    for (const RenderSample& s : samples) {
      for (std::size_t i = 0; i < s.fine_t.size(); i += 7) {
        if (s.fine_t[i] > 1e-4f) t.push_back(s.fine_t[i]);
        if (s.fine_d[i] > 1e-5f) d.push_back(s.fine_d[i]);
      }
    }
    const auto pct = [](std::vector<float>& v) {
      if (v.empty()) return 1.f;
      const auto k = static_cast<std::ptrdiff_t>(static_cast<double>(v.size() - 1) * 0.995);
      std::nth_element(v.begin(), v.begin() + k, v.end());
      return std::max(v[static_cast<std::size_t>(k)], 1e-4f);
    };
    m.render_scale = {pct(t), pct(d)};
  }
  std::vector<State> states;
  states.reserve(samples.size());
  for (const RenderSample& s : samples) states.push_back(sample_state(m, s));
  const RenderLayout L = render_layout(m.h);
  const int H = m.h.render_hidden;
  const int threads = thread_count(o.threads);
  Adam adam(L.size);
  double psnr = 0;
  for (int it = 0; it < o.iterations; ++it) {
    std::vector<Vec> grads(sz(threads), Vec(L.size, 0.f));
    std::vector<double> sse(sz(threads), 0.0);
    {
      std::vector<std::jthread> pool;
      for (int t = 0; t < threads; ++t) {
        pool.emplace_back([&, t] {
          std::mt19937_64 rng(o.seed * 7919ULL + static_cast<std::uint64_t>(it) * 131ULL + static_cast<std::uint64_t>(t));
          std::array<float, kRenderIn> f{};
          Vec h1(sz(H)), h2(sz(H)), g1(sz(H)), g2(sz(H));
          const float* w = m.render_w.data();
          float* g = grads[sz(t)].data();
          for (int b = t; b < o.batch_frames; b += threads) {
            const std::size_t si = rng() % samples.size();
            const RenderSample& s = samples[si];
            const State& st = states[si];
            const int S = s.size;
            for (int p = 0; p < o.pixels; ++p) {
              // half the pixels where something is visible, half anywhere
              int x = static_cast<int>(rng() % static_cast<std::uint64_t>(S)), y = static_cast<int>(rng() % static_cast<std::uint64_t>(S));
              if (p & 1) {
                for (int tries = 0; tries < 8; ++tries) {
                  if (s.fine_t[sz(y) * sz(S) + sz(x)] > 0.01f * m.render_scale[0] || s.fine_d[sz(y) * sz(S) + sz(x)] > 0.01f * m.render_scale[1]) break;
                  x = static_cast<int>(rng() % static_cast<std::uint64_t>(S));
                  y = static_cast<int>(rng() % static_cast<std::uint64_t>(S));
                }
              }
              render_features(m, st, x, y, f);
              for (int j = 0; j < H; ++j) h1[sz(j)] = std::max(0.f, w[L.b1 + sz(j)] + dot(w + L.w1 + sz(j) * kRenderIn, f.data(), kRenderIn));
              for (int j = 0; j < H; ++j) h2[sz(j)] = std::max(0.f, w[L.b2 + sz(j)] + dot(w + L.w2 + sz(j) * sz(H), h1.data(), H));
              const std::uint8_t* target = s.rgba.data() + (sz(S - 1 - y) * sz(S) + sz(x)) * 4;
              std::fill(g2.begin(), g2.end(), 0.f);
              const float gate = render_gate(f[0], f[1]);
              for (int c = 0; c < 4; ++c) {
                const float out = (w[L.bo + sz(c)] + dot(w + L.wo + sz(c) * sz(H), h2.data(), H)) * gate;
                const float e = out - static_cast<float>(target[c]) / 255.f;
                sse[sz(t)] += static_cast<double>(e * e);
                const float ge = 2.f * e * gate / static_cast<float>(o.pixels * 4);
                g[L.bo + sz(c)] += ge;
                for (int j = 0; j < H; ++j) {
                  g[L.wo + sz(c) * sz(H) + sz(j)] += ge * h2[sz(j)];
                  g2[sz(j)] += ge * w[L.wo + sz(c) * sz(H) + sz(j)];
                }
              }
              std::fill(g1.begin(), g1.end(), 0.f);
              for (int j = 0; j < H; ++j) {
                if (h2[sz(j)] <= 0.f) continue;
                g[L.b2 + sz(j)] += g2[sz(j)];
                for (int i = 0; i < H; ++i) {
                  g[L.w2 + sz(j) * sz(H) + sz(i)] += g2[sz(j)] * h1[sz(i)];
                  g1[sz(i)] += g2[sz(j)] * w[L.w2 + sz(j) * sz(H) + sz(i)];
                }
              }
              for (int j = 0; j < H; ++j) {
                if (h1[sz(j)] <= 0.f) continue;
                g[L.b1 + sz(j)] += g1[sz(j)];
                for (int i = 0; i < kRenderIn; ++i) g[L.w1 + sz(j) * kRenderIn + sz(i)] += g1[sz(j)] * f[sz(i)];
              }
            }
          }
        });
      }
    }
    for (int t = 1; t < threads; ++t) {
      for (std::size_t j = 0; j < L.size; ++j) grads[0][j] += grads[sz(t)][j];
    }
    double total = 0;
    for (const double v : sse) total += v;
    const double mse = total / (static_cast<double>(o.batch_frames) * o.pixels * 4);
    for (float& g : grads[0]) g /= static_cast<float>(o.batch_frames);
    const float lr = o.lr * (0.05f + 0.95f * 0.5f * (1.f + std::cos(3.14159265f * static_cast<float>(it) / static_cast<float>(o.iterations))));
    adam.step(m.render_w, grads[0], lr);
    psnr = -10.0 * std::log10(std::max(mse, 1e-10));
    if (o.progress && (it % 250 == 0 || it == o.iterations - 1)) o.progress(it, psnr);
  }
  return psnr;
}

// --- detail calibration and start points ------------------------------------------------------------------------------

namespace {

Clip rollout_clip(const Model& m, State s, std::span<const float> controls, std::uint64_t seed, int frames) {
  Clip c;
  c.allocate(s.size, frames);
  c.fps = m.fps;
  std::vector<float> rgba(sz(s.size) * sz(s.size) * 4);
  for (int f = 0; f < frames; ++f) {
    step(m, s, controls, seed);
    render(m, s, rgba);
    auto out = c.frame(f);
    for (std::size_t i = 0; i < rgba.size(); ++i) out[i] = static_cast<std::uint8_t>(rgba[i] * 255.f + 0.5f);
  }
  return c;
}


// How far a rollout's frame statistics are from the true run's: detail spectrum, motion, and how much light and cover
// it has (log ratios of the means, so too dim and too bright cost the same).
double detail_score(const metrics::ClipStats& ref, const metrics::ClipStats& test) {
  const metrics::StatDistance d = metrics::distance(ref, test);
  const auto mean = [](const std::vector<double>& v) {
    double s = 0;
    for (const double x : v) s += x;
    return v.empty() ? 0.0 : s / static_cast<double>(v.size());
  };
  const auto log_ratio = [](double a, double b) { return std::abs(std::log(std::max(1e-6, a) / std::max(1e-6, b))); };
  const double em = mean(ref.emission);  // light only counts for effects that give it (smoke gives almost none)
  return d.spectrum_l1 + std::abs(std::log(std::max(1e-3, d.motion_ratio))) + (em > 1e-3 ? log_ratio(mean(test.emission), em) : 0.0) +
         log_ratio(mean(test.coverage), mean(ref.coverage));
}

}  // namespace

double calibrate_detail(Model& m, std::span<const sim::Params> runs, int warm, int skip, int frames) {
  // As deployed: the model starts from its start point nearest the run's controls, with a seed of its own; after `skip`
  // frames its statistics over `frames` frames are compared with a real run at those controls (another seed, warmed up
  // by `warm` frames).
  struct Case {
    sim::Params p;
    int start = 0;
    metrics::ClipStats ref;
  };
  if (m.starts.empty()) throw std::invalid_argument("rollout: calibrate_detail needs start points");
  std::vector<Case> cases;
  for (const sim::Params& run : runs) {
    sim::Params p = run;
    p.seed = run.seed * 7 + 12345;  // a real run the model has not seen
    sim::Fluid f(p);
    for (int i = 0; i < warm; ++i) f.step_frame();
    Clip ref;
    ref.allocate(p.size, frames);
    ref.fps = p.fps;
    for (int i = 0; i < frames; ++i) {
      f.step_frame();
      f.render(ref.frame(i));
    }
    Case c;
    c.p = run;
    c.ref = metrics::stats(ref);
    float best = std::numeric_limits<float>::infinity();
    const std::array<float, 3> ctl{run.intensity, run.wind, run.turbulence};
    for (std::size_t k = 0; k < m.starts.size(); ++k) {
      float d = 0.f;
      for (std::size_t q = 0; q < 3 && q < m.starts[k].controls.size(); ++q) d += (m.starts[k].controls[q] - ctl[q]) * (m.starts[k].controls[q] - ctl[q]);
      if (d < best) {
        best = d;
        c.start = static_cast<int>(k);
      }
    }
    cases.push_back(std::move(c));
  }
  // kappa: 1 / mean of the contrast curve over the flicker noise, so contrast moves material around but adds none.
  {
    double mean = 0;
    constexpr int n = 20000;
    for (int i = 0; i < n; ++i) {
      const float phi = noise_flicker(m.noise, 9, static_cast<float>(i % 137) * 0.7f, static_cast<float>(i / 137) * 0.9f, 0.3f);
      const float t = std::clamp((phi - m.detail.edge0) / (m.detail.edge1 - m.detail.edge0), 0.f, 1.f);
      mean += static_cast<double>(t * t * (3.f - 2.f * t));
    }
    m.detail.kappa = static_cast<float>(n / std::max(1e-6, mean));
  }
  double best = std::numeric_limits<double>::infinity();
  DetailSpec keep = m.detail;
  struct Cell {
    float contrast, swirl, grow;
  };
  std::vector<Cell> grid;
  for (const float contrast : {0.5f, 1.f}) {
    for (const float swirl : {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f}) {
      for (const float grow : {1.f, 1.5f, 2.f, 4.f}) grid.push_back({contrast, swirl, grow});
    }
  }
  std::vector<double> scores(grid.size(), 0.0);
  std::atomic<std::size_t> next{0};
  {
    std::vector<std::jthread> pool;
    for (int t = 0; t < thread_count(0); ++t) {
      pool.emplace_back([&] {
        for (std::size_t g; (g = next++) < grid.size();) {
          Model mm = m;
          mm.detail.contrast = grid[g].contrast;
          mm.detail.swirl = grid[g].swirl;
          mm.detail.grow = grid[g].grow;
          double s = 0;
          for (const Case& c : cases) {
            const std::vector<float> controls{c.p.intensity, c.p.wind, c.p.turbulence};
            const std::uint64_t seed = c.p.seed * 31 + 777;
            State st = start(mm, c.start, c.p.size, controls, seed);
            for (int i = 0; i < skip; ++i) step(mm, st, controls, seed);
            s += detail_score(c.ref, metrics::stats(rollout_clip(mm, st, controls, seed, frames)));
          }
          scores[g] = s / static_cast<double>(cases.size());
        }
      });
    }
  }
  for (std::size_t g = 0; g < grid.size(); ++g) {
    if (scores[g] < best) {
      best = scores[g];
      keep.contrast = grid[g].contrast;
      keep.swirl = grid[g].swirl;
      keep.grow = grid[g].grow;
    }
  }
  keep.kappa = m.detail.kappa;
  m.detail = keep;
  return best;
}

void choose_starts(Model& m, std::span<const Run> runs, int count, int frame) {
  if (runs.empty() || count < 1) throw std::invalid_argument("rollout: no runs to take start points from");
  // Farthest-point order in control space, starting from the run nearest the middle.
  std::vector<std::size_t> order;
  std::vector<double> dist(runs.size(), std::numeric_limits<double>::infinity());
  const auto d2 = [&](std::size_t a, const std::vector<float>& c) {
    const auto ca = runs[a].controls();
    double s = 0;
    for (std::size_t k = 0; k < c.size(); ++k) s += static_cast<double>((ca[k] - c[k]) * (ca[k] - c[k]));
    return s;
  };
  std::size_t first = 0;
  for (std::size_t i = 1; i < runs.size(); ++i) {
    if (d2(i, {0.5f, 0.5f, 0.5f}) < d2(first, {0.5f, 0.5f, 0.5f})) first = i;
  }
  order.push_back(first);
  while (order.size() < std::min(runs.size(), sz(count))) {
    const auto last = runs[order.back()].controls();
    std::size_t pick = 0;
    double far = -1;
    for (std::size_t i = 0; i < runs.size(); ++i) {
      dist[i] = std::min(dist[i], d2(i, last));
      if (dist[i] > far) {
        far = dist[i];
        pick = i;
      }
    }
    order.push_back(pick);
  }
  m.starts.clear();
  const int R = m.h.res;
  for (const std::size_t i : order) {
    const Run& r = runs[i];
    const int f = std::clamp(frame, 0, r.frames - 1);
    StartPoint sp;
    sp.controls = r.controls();
    sp.seed = r.p.seed;
    sp.time = static_cast<float>(f + 1) / r.p.fps;
    sp.coarse.assign(r.coarse.begin() + static_cast<std::ptrdiff_t>(sz(f) * sz(R) * sz(R) * kPhys),
                     r.coarse.begin() + static_cast<std::ptrdiff_t>(sz(f + 1) * sz(R) * sz(R) * kPhys));
    if (m.h.start_fine > 0) {  // re-simulate to the frame for the fine fields, block-averaged to start_fine
      sim::Fluid fl(r.p);
      for (int k = 0; k <= f; ++k) fl.step_frame();
      const sim::State st = fl.state();
      const int F = m.h.start_fine, k = st.n / F;
      sp.fine_t.assign(sz(F) * sz(F), 0.f);
      sp.fine_d.assign(sz(F) * sz(F), 0.f);
      for (int y = 0; y < st.n; ++y) {
        for (int x = 0; x < st.n; ++x) {
          sp.fine_t[sz(y / k) * sz(F) + sz(x / k)] += st.temp[sz(y) * sz(st.n) + sz(x)] / static_cast<float>(k * k);
          sp.fine_d[sz(y / k) * sz(F) + sz(x / k)] += st.soot[sz(y) * sz(st.n) + sz(x)] / static_cast<float>(k * k);
        }
      }
    }
    m.starts.push_back(std::move(sp));
  }
}

// --- recipe -------------------------------------------------------------------------------------------------------------

SimRecipe recipe_for(sim::Effect e) {
  SimRecipe r;
  r.effect = e;
  r.stepper.activity_stage = 800;
  r.stepper.activity = 100.f;
  if (e == sim::Effect::smoke) r.start_fine = 64;  // smoke lingers: growing its detail would take seconds
  if (e == sim::Effect::explosion) {
    r.runs = 240;
    r.frames = 90;
    r.start_frame = 0;
    r.start_fine = 64;  // the burst's shape is the look
    r.starts = 16;
    r.stepper.burn_max = 32;
  }
  return r;
}

sim::Params recipe_run(const SimRecipe& r, std::uint64_t index) {
  std::mt19937_64 rng(index * 0x9E3779B97F4A7C15ULL + r.salt);
  std::uniform_real_distribution<float> u(0.f, 1.f);
  sim::Params p;
  p.effect = r.effect;
  p.intensity = u(rng);
  p.wind = u(rng);
  p.turbulence = u(rng);
  p.seed = 500000 + r.salt * 100000 + index;
  p.size = 128;
  return p;
}

std::vector<Run> record_runs(const SimRecipe& r) {
  std::vector<Run> runs(sz(r.runs));
  std::atomic<int> next{0};
  {
    std::vector<std::jthread> pool;
    for (int t = 0; t < thread_count(r.threads); ++t) {
      pool.emplace_back([&] {
        for (int i; (i = next++) < r.runs;) runs[sz(i)] = record_run(recipe_run(r, static_cast<std::uint64_t>(i)), r.frames, 32);
      });
    }
  }
  return runs;
}

Model recipe_model(const SimRecipe& r) {
  Hyper h;
  h.res = 32;
  h.n_controls = sim::kControls;
  const bool one_shot = !sim::effect_loops(r.effect);
  h.n_age = one_shot ? 2 : 0;
  h.frames = one_shot ? r.frames : 0;
  h.start_fine = r.start_fine;
  Model m = init_model(h, 1);
  m.effect = std::string(sim::effect_name(r.effect));
  m.fps = 30.f;
  m.loop = !one_shot;
  m.control_names.assign(sim::kControlNames.begin(), sim::kControlNames.end());
  m.noise.flicker_rate = r.effect == sim::Effect::fire ? 2.6f : 1.4f;  // as the simulation's source flicker
  m.detail.swirl_control = 2;                                            // turbulence
  return m;
}

FinishResult finish_model(Model& m, const SimRecipe& r, std::span<const Run> runs) {
  FinishResult out;
  const bool one_shot = !sim::effect_loops(r.effect);
  const int threads = thread_count(r.threads);
  // renderer: two frames from each of the first render_runs runs, re-simulated at full size
  std::vector<RenderSample> samples;
  {
    const int n = std::min(static_cast<int>(runs.size()), r.render_runs);
    std::vector<std::vector<RenderSample>> per(sz(n));
    std::atomic<int> next{0};
    {
      std::vector<std::jthread> pool;
      for (int t = 0; t < threads; ++t) {
        pool.emplace_back([&] {
          for (int i; (i = next++) < n;) {
            sim::Fluid f(runs[sz(i)].p);
            const int span = std::max(1, r.frames - 20);
            const int a = 10 + (i * 37) % span, b = 10 + (i * 71 + 13) % span;
            for (int k = 0; k <= std::max(a, b); ++k) {
              f.step_frame();
              if (k == a || k == b) per[sz(i)].push_back(render_sample(f, m.h.res));
            }
          }
        });
      }
    }
    for (auto& v : per) {
      for (auto& s : v) samples.push_back(std::move(s));
    }
  }
  RendererOptions ro = r.renderer;
  ro.threads = r.threads;
  out.render_psnr = train_renderer(m, samples, ro);
  samples.clear();
  choose_starts(m, runs, r.starts, r.start_frame);
  quantise_like_storage(m);
  // detail layer, as deployed (from start points, new seeds), at the controls of four training runs
  std::vector<sim::Params> cal;
  for (std::size_t i = 0; i < std::min<std::size_t>(4, runs.size()); ++i) cal.push_back(runs[i].p);
  out.detail_score = one_shot ? calibrate_detail(m, cal, 1, 0, std::min(89, r.frames - 1)) : calibrate_detail(m, cal, 150, 60, 180);
  return out;
}

}  // namespace nfx::rollout

#ifdef NFX_ROLLOUT_PUSHED
#pragma GCC pop_options
#endif
