// Training study N's 3D rollout effect (include/neuralfx/rollout3d_train.hpp). The forward pass mirrors coarse_step() in
// src/core/rollout3d.cpp operation by operation; tests/test_3d.cpp checks it against that reference and the gradients
// against finite differences.
#include <neuralfx/binio.hpp>
#include <neuralfx/rollout3d_train.hpp>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <random>
#include <stdexcept>

// AVX2 + FMA for the loops below (the trainer needs x86-64-v3, as study D's).
#if defined(__GNUC__) && !defined(__clang__) && defined(__x86_64__)
#pragma GCC push_options
#pragma GCC target("arch=x86-64-v3")
#define NFX_ROLLOUT3D_PUSHED 1
#endif

namespace nfx::rollout3d {

namespace {

using Vec = std::vector<float>;
std::size_t zu(int v) { return static_cast<std::size_t>(v); }
float fl(int v) { return static_cast<float>(v); }

struct Grid {
  int rx, ry, rz;
  std::size_t at(int x, int y, int z) const { return (zu(y) * zu(rz) + zu(z)) * zu(rx) + zu(x); }
  int n() const { return rx * ry * rz; }
};

// --- operations on [cell][channel] layouts, with their transposes ------------------------------------------------------

void conv3(const Grid& g, const float* in, int ci, const float* W, const float* b, int co, float* out) {
  for (int y = 0; y < g.ry; ++y) {
    for (int z = 0; z < g.rz; ++z) {
      for (int x = 0; x < g.rx; ++x) {
        float* __restrict o = out + g.at(x, y, z) * zu(co);
        for (int k = 0; k < co; ++k) o[k] = b[k];
        for (int dy = -1; dy <= 1; ++dy) {
          const int yy = y + dy;
          if (yy < 0 || yy >= g.ry) continue;
          for (int dz = -1; dz <= 1; ++dz) {
            const int zz = z + dz;
            if (zz < 0 || zz >= g.rz) continue;
            for (int dx = -1; dx <= 1; ++dx) {
              const int xx = x + dx;
              if (xx < 0 || xx >= g.rx) continue;
              const float* a = in + g.at(xx, yy, zz) * zu(ci);
              const float* w = W + zu(((dy + 1) * 3 + (dz + 1)) * 3 + (dx + 1)) * zu(ci) * zu(co);
              for (int c = 0; c < ci; ++c) {
                const float av = a[c];
                const float* __restrict wr = w + zu(c) * zu(co);
                for (int k = 0; k < co; ++k) o[k] += wr[k] * av;
              }
            }
          }
        }
      }
    }
  }
}

void conv3_back(const Grid& g, const float* in, int ci, const float* W, int co, const float* go_all, float* gin, float* gW, float* gb) {
  for (int y = 0; y < g.ry; ++y) {
    for (int z = 0; z < g.rz; ++z) {
      for (int x = 0; x < g.rx; ++x) {
        const float* go = go_all + g.at(x, y, z) * zu(co);
        for (int k = 0; k < co; ++k) gb[k] += go[k];
        for (int dy = -1; dy <= 1; ++dy) {
          const int yy = y + dy;
          if (yy < 0 || yy >= g.ry) continue;
          for (int dz = -1; dz <= 1; ++dz) {
            const int zz = z + dz;
            if (zz < 0 || zz >= g.rz) continue;
            for (int dx = -1; dx <= 1; ++dx) {
              const int xx = x + dx;
              if (xx < 0 || xx >= g.rx) continue;
              const std::size_t nb = g.at(xx, yy, zz);
              const float* a = in + nb * zu(ci);
              float* ga = gin ? gin + nb * zu(ci) : nullptr;
              const std::size_t t = zu(((dy + 1) * 3 + (dz + 1)) * 3 + (dx + 1)) * zu(ci) * zu(co);
              for (int c = 0; c < ci; ++c) {
                const float av = a[c];
                float* __restrict gw = gW + t + zu(c) * zu(co);
                const float* __restrict wr = W + t + zu(c) * zu(co);
                float s = 0.f;
                for (int k = 0; k < co; ++k) {
                  gw[k] += av * go[k];
                  s += wr[k] * go[k];
                }
                if (ga) ga[c] += s;
              }
            }
          }
        }
      }
    }
  }
}

void conv1(int n, const float* in, int ci, const float* W, const float* b, int co, float* out) {
  for (int i = 0; i < n; ++i) {
    float* __restrict o = out + zu(i) * zu(co);
    for (int k = 0; k < co; ++k) o[k] = b[k];
    const float* a = in + zu(i) * zu(ci);
    for (int c = 0; c < ci; ++c) {
      const float av = a[c];
      for (int k = 0; k < co; ++k) o[k] += W[zu(c) * zu(co) + zu(k)] * av;
    }
  }
}

void conv1_back(int n, const float* in, int ci, const float* W, int co, const float* g, float* gin, float* gW, float* gb) {
  for (int i = 0; i < n; ++i) {
    const float* go = g + zu(i) * zu(co);
    const float* a = in + zu(i) * zu(ci);
    float* ga = gin + zu(i) * zu(ci);
    for (int k = 0; k < co; ++k) gb[k] += go[k];
    for (int c = 0; c < ci; ++c) {
      float s = 0.f;
      for (int k = 0; k < co; ++k) {
        gW[zu(c) * zu(co) + zu(k)] += a[c] * go[k];
        s += W[zu(c) * zu(co) + zu(k)] * go[k];
      }
      ga[c] += s;
    }
  }
}

int cl(int v, int n) { return v < 0 ? 0 : (v >= n ? n - 1 : v); }

void divergence(const Grid& g, const float* S, int C, const float* d, int O, float qs, float* div) {
  const auto U = [&](int x, int y, int z, int c) { return S[g.at(cl(x, g.rx), cl(y, g.ry), cl(z, g.rz)) * zu(C) + zu(c)]; };
  for (int y = 0; y < g.ry; ++y) {
    for (int z = 0; z < g.rz; ++z) {
      for (int x = 0; x < g.rx; ++x) {
        div[g.at(x, y, z)] = -0.5f * (U(x + 1, y, z, 0) - U(x - 1, y, z, 0) + U(x, y + 1, z, 1) - U(x, y - 1, z, 1) + U(x, y, z + 1, 2) - U(x, y, z - 1, 2)) +
                             qs * d[g.at(x, y, z) * zu(O) + zu(O - 1)];
      }
    }
  }
}
void divergence_back(const Grid& g, int C, int O, float qs, const float* gdiv, float* gS, float* gd) {
  for (int y = 0; y < g.ry; ++y) {
    for (int z = 0; z < g.rz; ++z) {
      for (int x = 0; x < g.rx; ++x) {
        const float v = gdiv[g.at(x, y, z)];
        gd[g.at(x, y, z) * zu(O) + zu(O - 1)] += qs * v;
        gS[g.at(cl(x + 1, g.rx), y, z) * zu(C)] -= 0.5f * v;
        gS[g.at(cl(x - 1, g.rx), y, z) * zu(C)] += 0.5f * v;
        gS[g.at(x, cl(y + 1, g.ry), z) * zu(C) + 1] -= 0.5f * v;
        gS[g.at(x, cl(y - 1, g.ry), z) * zu(C) + 1] += 0.5f * v;
        gS[g.at(x, y, cl(z + 1, g.rz)) * zu(C) + 2] -= 0.5f * v;
        gS[g.at(x, y, cl(z - 1, g.rz)) * zu(C) + 2] += 0.5f * v;
      }
    }
  }
}

// The pressure's neighbours: zero outside the sides and the top; below the floor, the floor cell itself. The operator
// is symmetric (the floor's own value sits on the diagonal), so the Jacobi iteration is its own transpose.
void neighbour_sum(const Grid& g, const float* p, float* o) {
  for (int y = 0; y < g.ry; ++y) {
    for (int z = 0; z < g.rz; ++z) {
      for (int x = 0; x < g.rx; ++x) {
        float s = 0.f;
        if (x > 0) s += p[g.at(x - 1, y, z)];
        if (x < g.rx - 1) s += p[g.at(x + 1, y, z)];
        if (z > 0) s += p[g.at(x, y, z - 1)];
        if (z < g.rz - 1) s += p[g.at(x, y, z + 1)];
        s += p[g.at(x, std::max(y - 1, 0), z)];
        if (y < g.ry - 1) s += p[g.at(x, y + 1, z)];
        o[g.at(x, y, z)] = s;
      }
    }
  }
}
void jacobi(const Grid& g, int K, const float* div, float* p, float* tmp) {
  const int n = g.n();
  for (int k = 0; k < K; ++k) {
    neighbour_sum(g, p, tmp);
    for (int i = 0; i < n; ++i) p[i] = (1.f / 6.f) * (div[i] + tmp[i]);
  }
}
void jacobi_back(const Grid& g, int K, const float* gp, float* gdiv, float* gp0, float* w, float* tmp) {
  const int n = g.n();
  std::memcpy(w, gp, sizeof(float) * zu(n));
  std::fill_n(gdiv, zu(n), 0.f);
  for (int k = 0; k < K; ++k) {
    for (int i = 0; i < n; ++i) gdiv[i] += (1.f / 6.f) * w[i];
    neighbour_sum(g, w, tmp);
    for (int i = 0; i < n; ++i) w[i] = (1.f / 6.f) * tmp[i];
  }
  for (int i = 0; i < n; ++i) gp0[i] += w[i];
}

float pval(const Grid& g, const float* p, int x, int y, int z) {
  if (x < 0 || z < 0 || x >= g.rx || z >= g.rz || y >= g.ry) return 0.f;
  return p[g.at(x, std::max(y, 0), z)];
}
void subtract_grad(const Grid& g, const float* p, float* S, int C) {
  for (int y = 0; y < g.ry; ++y) {
    for (int z = 0; z < g.rz; ++z) {
      for (int x = 0; x < g.rx; ++x) {
        float* s = S + g.at(x, y, z) * zu(C);
        s[0] -= 0.5f * (pval(g, p, x + 1, y, z) - pval(g, p, x - 1, y, z));
        s[1] -= 0.5f * (pval(g, p, x, y + 1, z) - pval(g, p, x, y - 1, z));
        s[2] -= 0.5f * (pval(g, p, x, y, z + 1) - pval(g, p, x, y, z - 1));
      }
    }
  }
}
void subtract_grad_back(const Grid& g, const float* gS, int C, float* gp) {
  const auto add = [&](int x, int y, int z, float v) {
    if (x < 0 || z < 0 || x >= g.rx || z >= g.rz || y >= g.ry) return;
    gp[g.at(x, std::max(y, 0), z)] += v;
  };
  for (int y = 0; y < g.ry; ++y) {
    for (int z = 0; z < g.rz; ++z) {
      for (int x = 0; x < g.rx; ++x) {
        const float* s = gS + g.at(x, y, z) * zu(C);
        add(x + 1, y, z, -0.5f * s[0]);
        add(x - 1, y, z, 0.5f * s[0]);
        add(x, y + 1, z, -0.5f * s[1]);
        add(x, y - 1, z, 0.5f * s[1]);
        add(x, y, z + 1, -0.5f * s[2]);
        add(x, y, z - 1, 0.5f * s[2]);
      }
    }
  }
}

struct Tri {
  std::size_t c[8];  // corner cells: bit 0 x, bit 1 z, bit 2 y
  float fx, fy, fz;
  bool inx, iny, inz;  // the back-traced position is inside the grid (not clamped) along each axis
};
Tri tri_at(const Grid& g, float rxp, float ryp, float rzp) {
  Tri t{};
  const float px = std::clamp(rxp, 0.f, fl(g.rx - 1)), py = std::clamp(ryp, 0.f, fl(g.ry - 1)), pz = std::clamp(rzp, 0.f, fl(g.rz - 1));
  t.inx = rxp > 0.f && rxp < fl(g.rx - 1);
  t.iny = ryp > 0.f && ryp < fl(g.ry - 1);
  t.inz = rzp > 0.f && rzp < fl(g.rz - 1);
  const int x0 = std::min(static_cast<int>(px), g.rx - 2), y0 = std::min(static_cast<int>(py), g.ry - 2), z0 = std::min(static_cast<int>(pz), g.rz - 2);
  t.fx = px - fl(x0);
  t.fy = py - fl(y0);
  t.fz = pz - fl(z0);
  for (int k = 0; k < 8; ++k) t.c[k] = g.at(x0 + (k & 1), y0 + ((k >> 2) & 1), z0 + ((k >> 1) & 1));
  return t;
}

// The same arithmetic as rollout3d.cpp's trilinear(): x, then z, then y.
float tri_value(const Tri& t, const float* S, int C, int c) {
  const auto v = [&](int k) { return S[t.c[k] * zu(C) + zu(c)]; };
  const float a00 = v(0) + t.fx * (v(1) - v(0)), a10 = v(2) + t.fx * (v(3) - v(2));
  const float a01 = v(4) + t.fx * (v(5) - v(4)), a11 = v(6) + t.fx * (v(7) - v(6));
  const float b0 = a00 + t.fz * (a10 - a00), b1 = a01 + t.fz * (a11 - a01);
  return b0 + t.fy * (b1 - b0);
}

void advect(const Grid& g, const float* S, int C, float* out) {
  for (int y = 0; y < g.ry; ++y) {
    for (int z = 0; z < g.rz; ++z) {
      for (int x = 0; x < g.rx; ++x) {
        const float* s = S + g.at(x, y, z) * zu(C);
        const Tri t = tri_at(g, fl(x) - s[0], fl(y) - s[1], fl(z) - s[2]);
        float* o = out + g.at(x, y, z) * zu(C);
        for (int c = 0; c < C; ++c) o[c] = tri_value(t, S, C, c);
      }
    }
  }
}
void advect_back(const Grid& g, const float* S, int C, const float* gout, float* gS) {
  for (int y = 0; y < g.ry; ++y) {
    for (int z = 0; z < g.rz; ++z) {
      for (int x = 0; x < g.rx; ++x) {
        const std::size_t i = g.at(x, y, z);
        const float* s = S + i * zu(C);
        const Tri t = tri_at(g, fl(x) - s[0], fl(y) - s[1], fl(z) - s[2]);
        const float fx = t.fx, fy = t.fy, fz = t.fz;
        const float w[8] = {(1 - fx) * (1 - fz) * (1 - fy), fx * (1 - fz) * (1 - fy), (1 - fx) * fz * (1 - fy), fx * fz * (1 - fy),
                            (1 - fx) * (1 - fz) * fy,       fx * (1 - fz) * fy,       (1 - fx) * fz * fy,       fx * fz * fy};
        const float* go = gout + i * zu(C);
        float dpx = 0.f, dpy = 0.f, dpz = 0.f;
        for (int c = 0; c < C; ++c) {
          const float gc = go[c];
          if (gc == 0.f) continue;
          float v[8];
          for (int k = 0; k < 8; ++k) {
            v[k] = S[t.c[k] * zu(C) + zu(c)];
            gS[t.c[k] * zu(C) + zu(c)] += gc * w[k];
          }
          // derivatives of the trilinear value along x, z and y
          dpx += gc * ((1 - fz) * (1 - fy) * (v[1] - v[0]) + fz * (1 - fy) * (v[3] - v[2]) + (1 - fz) * fy * (v[5] - v[4]) + fz * fy * (v[7] - v[6]));
          dpz += gc * ((1 - fx) * (1 - fy) * (v[2] - v[0]) + fx * (1 - fy) * (v[3] - v[1]) + (1 - fx) * fy * (v[6] - v[4]) + fx * fy * (v[7] - v[5]));
          dpy += gc * ((1 - fx) * (1 - fz) * (v[4] - v[0]) + fx * (1 - fz) * (v[5] - v[1]) + (1 - fx) * fz * (v[6] - v[2]) + fx * fz * (v[7] - v[3]));
        }
        if (t.inx) gS[i * zu(C)] -= dpx;
        if (t.iny) gS[i * zu(C) + 1] -= dpy;
        if (t.inz) gS[i * zu(C) + 2] -= dpz;
      }
    }
  }
}

// --- one differentiable coarse step -------------------------------------------------------------------------------------

struct Cache {
  Vec X, y1, a1, y2, a2, out, mid, div, p, proj, next;
  std::array<float, 128> g1{}, e1{}, g2{}, e2{};
  std::vector<std::uint8_t> clamped;
};

void film(const Model& m, const StepLayout& L, std::span<const float> cond, Cache& c) {
  const int H = m.h.hidden;
  for (int j = 0; j < H; ++j) {
    c.g1[zu(j)] = 1.f;
    c.e1[zu(j)] = 0.f;
    c.g2[zu(j)] = 1.f;
    c.e2[zu(j)] = 0.f;
    for (int k = 0; k < m.h.cond(); ++k) {
      const float v = cond[zu(k)];
      c.g1[zu(j)] += m.step_w[L.g1 + zu(k) * zu(H) + zu(j)] * v;
      c.e1[zu(j)] += m.step_w[L.e1 + zu(k) * zu(H) + zu(j)] * v;
      c.g2[zu(j)] += m.step_w[L.g2 + zu(k) * zu(H) + zu(j)] * v;
      c.e2[zu(j)] += m.step_w[L.e2 + zu(k) * zu(H) + zu(j)] * v;
    }
  }
}

// in -> c.next; pressure: warm start in, solution out (also kept in c.p).
void forward(const Model& m, const StepLayout& L, const float* in, const float* noise, std::span<const float> cond, Vec& pressure, Cache& c, Vec& tmp) {
  const Hyper& h = m.h;
  const Grid g{h.rx, h.ry, h.rz};
  const int N = g.n(), C = h.channels(), I = h.inputs(), H = h.hidden, O = h.outputs();
  const float* w = m.step_w.data();
  c.X.resize(zu(N) * zu(I));
  for (int y = 0; y < g.ry; ++y) {
    for (int z = 0; z < g.rz; ++z) {
      for (int x = 0; x < g.rx; ++x) {
        const std::size_t i = g.at(x, y, z);
        float* xi = c.X.data() + i * zu(I);
        for (int k = 0; k < C; ++k) xi[k] = in[i * zu(C) + zu(k)] / (k < kPhys ? m.scale[zu(k)] : 1.f);
        for (int k = 0; k < kNoise; ++k) xi[C + k] = noise[i * kNoise + zu(k)];
        xi[C + kNoise] = (fl(x) + 0.5f) / fl(g.rx) * 2.f - 1.f;
        xi[C + kNoise + 1] = (fl(y) + 0.5f) / fl(g.ry) * 2.f - 1.f;
        xi[C + kNoise + 2] = (fl(z) + 0.5f) / fl(g.rz) * 2.f - 1.f;
      }
    }
  }
  film(m, L, cond, c);
  c.y1.resize(zu(N) * zu(H));
  c.a1.resize(zu(N) * zu(H));
  conv3(g, c.X.data(), I, w + L.w1, w + L.b1, H, c.y1.data());
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < H; ++j) c.a1[zu(i) * zu(H) + zu(j)] = std::max(0.f, c.g1[zu(j)] * c.y1[zu(i) * zu(H) + zu(j)] + c.e1[zu(j)]);
  }
  c.y2.resize(zu(N) * zu(H));
  c.a2.resize(zu(N) * zu(H));
  conv3(g, c.a1.data(), H, w + L.w2, w + L.b2, H, c.y2.data());
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < H; ++j) c.a2[zu(i) * zu(H) + zu(j)] = std::max(0.f, c.g2[zu(j)] * c.y2[zu(i) * zu(H) + zu(j)] + c.e2[zu(j)]);
  }
  c.out.resize(zu(N) * zu(O));
  conv1(N, c.a2.data(), H, w + L.wo, w + L.bo, O, c.out.data());
  c.mid.resize(zu(N) * zu(C));
  for (int i = 0; i < N; ++i) {
    for (int k = 0; k < kPhys; ++k) c.mid[zu(i) * zu(C) + zu(k)] = in[zu(i) * zu(C) + zu(k)] + m.scale[zu(k)] * c.out[zu(i) * zu(O) + zu(k)];
    for (int k = kPhys; k < C; ++k) c.mid[zu(i) * zu(C) + zu(k)] = std::tanh(in[zu(i) * zu(C) + zu(k)] + c.out[zu(i) * zu(O) + zu(k)]);
  }
  c.div.resize(zu(N));
  divergence(g, c.mid.data(), C, c.out.data(), O, m.qscale, c.div.data());
  tmp.resize(zu(N));
  jacobi(g, h.jacobi, c.div.data(), pressure.data(), tmp.data());
  c.p = pressure;
  c.proj = c.mid;
  subtract_grad(g, c.p.data(), c.proj.data(), C);
  c.next.resize(zu(N) * zu(C));
  advect(g, c.proj.data(), C, c.next.data());
  c.clamped.assign(zu(N) * kPhys, 0);
  for (int i = 0; i < N; ++i) {
    for (int k = 0; k < kPhys; ++k) {
      float& v = c.next[zu(i) * zu(C) + zu(k)];
      if (v < m.lo[zu(k)] || v > m.hi[zu(k)]) {
        c.clamped[zu(i) * kPhys + zu(k)] = 1;
        v = std::clamp(v, m.lo[zu(k)], m.hi[zu(k)]);
      }
    }
  }
}

// gS (wrt c.next, consumed), gp_out (wrt c.p) -> gS_in, gp_in (wrt the warm start); grad accumulated.
void backward(const Model& m, const StepLayout& L, const Cache& c, std::span<const float> cond, Vec& gS, const Vec& gp_out, Vec& gS_in, Vec& gp_in,
              float* grad) {
  const Hyper& h = m.h;
  const Grid g{h.rx, h.ry, h.rz};
  const int N = g.n(), C = h.channels(), I = h.inputs(), H = h.hidden, O = h.outputs();
  const float* w = m.step_w.data();
  for (int i = 0; i < N; ++i) {
    for (int k = 0; k < kPhys; ++k) {
      if (c.clamped[zu(i) * kPhys + zu(k)]) gS[zu(i) * zu(C) + zu(k)] = 0.f;
    }
  }
  Vec gproj(zu(N) * zu(C), 0.f);
  advect_back(g, c.proj.data(), C, gS.data(), gproj.data());
  Vec gp(gp_out);
  subtract_grad_back(g, gproj.data(), C, gp.data());
  Vec gdiv(zu(N)), wk(zu(N)), tmp(zu(N));
  gp_in.assign(zu(N), 0.f);
  jacobi_back(g, h.jacobi, gp.data(), gdiv.data(), gp_in.data(), wk.data(), tmp.data());
  Vec& gmid = gproj;
  Vec gout(zu(N) * zu(O), 0.f);
  divergence_back(g, C, O, m.qscale, gdiv.data(), gmid.data(), gout.data());
  for (int i = 0; i < N; ++i) {
    for (int k = kPhys; k < C; ++k) {
      const float t = c.mid[zu(i) * zu(C) + zu(k)];
      gmid[zu(i) * zu(C) + zu(k)] *= 1.f - t * t;
    }
    for (int k = 0; k < C; ++k) gout[zu(i) * zu(O) + zu(k)] += (k < kPhys ? m.scale[zu(k)] : 1.f) * gmid[zu(i) * zu(C) + zu(k)];
  }
  gS_in = gmid;
  Vec ga2(zu(N) * zu(H), 0.f);
  conv1_back(N, c.a2.data(), H, w + L.wo, O, gout.data(), ga2.data(), grad + L.wo, grad + L.bo);
  const auto film_back = [&](const Vec& a, const Vec& y, const std::array<float, 128>& gam, Vec& ga, std::size_t G, std::size_t E) {
    std::vector<double> sg(zu(H), 0.0), se(zu(H), 0.0);
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < H; ++j) {
        const std::size_t q = zu(i) * zu(H) + zu(j);
        const float gz = a[q] > 0.f ? ga[q] : 0.f;
        sg[zu(j)] += static_cast<double>(gz * y[q]);
        se[zu(j)] += static_cast<double>(gz);
        ga[q] = gz * gam[zu(j)];
      }
    }
    for (int k = 0; k < h.cond(); ++k) {
      for (int j = 0; j < H; ++j) {
        grad[G + zu(k) * zu(H) + zu(j)] += static_cast<float>(sg[zu(j)]) * cond[zu(k)];
        grad[E + zu(k) * zu(H) + zu(j)] += static_cast<float>(se[zu(j)]) * cond[zu(k)];
      }
    }
  };
  film_back(c.a2, c.y2, c.g2, ga2, L.g2, L.e2);
  Vec ga1(zu(N) * zu(H), 0.f);
  conv3_back(g, c.a1.data(), H, w + L.w2, H, ga2.data(), ga1.data(), grad + L.w2, grad + L.b2);
  film_back(c.a1, c.y1, c.g1, ga1, L.g1, L.e1);
  Vec gX(zu(N) * zu(I), 0.f);
  conv3_back(g, c.X.data(), I, w + L.w1, H, ga1.data(), gX.data(), grad + L.w1, grad + L.b1);
  for (int i = 0; i < N; ++i) {
    for (int k = 0; k < C; ++k) gS_in[zu(i) * zu(C) + zu(k)] += gX[zu(i) * zu(I) + zu(k)] / (k < kPhys ? m.scale[zu(k)] : 1.f);
  }
}

void state_of(const Model& m, const Run& r, int i, Vec& S, Vec& phys) {
  const int N = m.h.cells(), C = m.h.channels();
  phys.resize(zu(N) * kPhys);
  r.state(i, zu(N) * kPhys, phys);
  S.assign(zu(N) * zu(C), 0.f);
  for (int j = 0; j < N; ++j) {
    for (int k = 0; k < kPhys; ++k) S[zu(j) * zu(C) + zu(k)] = phys[zu(j) * kPhys + zu(k)];
  }
}

}  // namespace

// --- data ---------------------------------------------------------------------------------------------------------------

void coarse_from_sim(const sim3d::Fluid& f, const Hyper& h, std::span<float> out) {
  if (f.nx() != h.fx() || f.ny() != h.fy() || f.nz() != h.fz()) throw std::invalid_argument("rollout3d: the simulation's grid is not the model's fine grid");
  const std::size_t n = h.fine_cells();
  std::vector<float> T(n), D(n), uvw(3 * n), u(n), v(n), w(n);
  f.fields(T, D);
  f.velocity(uvw);
  for (std::size_t i = 0; i < n; ++i) {
    u[i] = uvw[3 * i];
    v[i] = uvw[3 * i + 1];
    w[i] = uvw[3 * i + 2];
  }
  block_average(h, u, out, 0, kPhys);
  block_average(h, v, out, 1, kPhys);
  block_average(h, w, out, 2, kPhys);
  block_average(h, T, out, 3, kPhys);
  block_average(h, D, out, 4, kPhys);
  const float k = 1.f / (fl(h.fine) * f.params().fps);  // fine cells per second -> coarse cells per frame
  for (int i = 0; i < h.cells(); ++i) {
    for (int c = 0; c < 3; ++c) out[zu(i) * kPhys + zu(c)] *= k;
  }
}

Run record_run(const sim3d::Params& p, int frames, const Hyper& h) {
  Run r;
  r.p = p;
  r.frames = frames;
  const std::size_t per = zu(h.cells()) * kPhys;
  r.coarse.resize(zu(frames) * per);
  sim3d::Fluid f(p);
  std::vector<float> c(per);
  for (int i = 0; i < frames; ++i) {
    f.step_frame();
    coarse_from_sim(f, h, c);
    for (std::size_t j = 0; j < per; ++j) r.coarse[zu(i) * per + j] = static_cast<std::float16_t>(c[j]);
  }
  return r;
}

namespace {
constexpr char kRunMagic[8] = {'N', 'V', '3', 'R', 'U', 'N', '0', '1'};
constexpr char kCkptMagic[8] = {'N', 'V', '3', 'C', 'K', 'P', 'T', '1'};
}  // namespace

std::expected<void, std::string> save_run(const std::filesystem::path& path, const Run& r) {
  const std::filesystem::path tmp = path.string() + ".tmp";
  {
    std::ofstream o(tmp, std::ios::binary);
    if (!o) return std::unexpected("cannot write " + tmp.string());
    o.write(kRunMagic, 8);
    bin::put(o, static_cast<std::int32_t>(r.p.effect == sim3d::Effect::smoke ? 1 : 0));
    for (const float v : {r.p.intensity, r.p.wind, r.p.turbulence, r.p.fps}) bin::put(o, v);
    bin::put(o, r.p.seed);
    for (const int v : {r.p.width, r.p.height, r.p.substeps, r.p.pressure_iters, r.frames}) bin::put(o, static_cast<std::int32_t>(v));
    bin::put(o, static_cast<std::uint64_t>(r.coarse.size()));
    for (const std::float16_t v : r.coarse) bin::put(o, std::bit_cast<std::uint16_t>(v));
    if (!o) return std::unexpected("write failed: " + tmp.string());
  }
  std::filesystem::rename(tmp, path);
  return {};
}

std::expected<Run, std::string> load_run(const std::filesystem::path& path) {
  std::ifstream i(path, std::ios::binary);
  if (!i) return std::unexpected("cannot open " + path.string());
  char magic[8];
  i.read(magic, 8);
  if (!i || std::memcmp(magic, kRunMagic, 8) != 0) return std::unexpected("not a 3D run: " + path.string());
  Run r;
  auto e = bin::get<std::int32_t>(i);
  if (!e) return std::unexpected(e.error());
  r.p.effect = *e == 1 ? sim3d::Effect::smoke : sim3d::Effect::fire;
  for (float* v : {&r.p.intensity, &r.p.wind, &r.p.turbulence, &r.p.fps}) {
    auto x = bin::get<float>(i);
    if (!x) return std::unexpected(x.error());
    *v = *x;
  }
  auto seed = bin::get<std::uint64_t>(i);
  if (!seed) return std::unexpected(seed.error());
  r.p.seed = *seed;
  for (int* v : {&r.p.width, &r.p.height, &r.p.substeps, &r.p.pressure_iters, &r.frames}) {
    auto x = bin::get<std::int32_t>(i);
    if (!x) return std::unexpected(x.error());
    *v = *x;
  }
  auto n = bin::get<std::uint64_t>(i);
  if (!n || *n > (std::uint64_t{1} << 32)) return std::unexpected("bad run size");
  r.coarse.resize(*n);
  std::vector<std::uint16_t> raw(*n);
  i.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size() * 2));
  if (!i) return std::unexpected("truncated run: " + path.string());
  for (std::size_t k = 0; k < raw.size(); ++k) r.coarse[k] = std::bit_cast<std::float16_t>(raw[k]);
  return r;
}

void set_normalisation(Model& m, std::span<const Run> runs) {
  std::array<double, kPhys> s2{};
  std::array<float, kPhys> mx{};
  std::size_t count = 0;
  for (const Run& r : runs) {
    for (std::size_t j = 0; j < r.coarse.size(); j += kPhys) {
      for (int k = 0; k < kPhys; ++k) {
        const float v = static_cast<float>(r.coarse[j + zu(k)]);
        s2[zu(k)] += static_cast<double>(v) * v;
        mx[zu(k)] = std::max(mx[zu(k)], std::abs(v));
      }
      ++count;
    }
  }
  for (int k = 0; k < kPhys; ++k) {
    m.scale[zu(k)] = static_cast<float>(std::sqrt(s2[zu(k)] / static_cast<double>(std::max<std::size_t>(1, count)))) + 1e-4f;
    m.hi[zu(k)] = 1.5f * mx[zu(k)] + 1e-4f;
    m.lo[zu(k)] = k < 3 ? -m.hi[zu(k)] : 0.f;
  }
}

// --- the stepper -----------------------------------------------------------------------------------------------------

double window_loss(const Model& m, const Run& r, int first, int unroll, int burn, float sigma, float profile, std::uint64_t noise_seed,
                   std::vector<float>* grad, float activity) {
  const Hyper& h = m.h;
  const StepLayout L = step_layout(h);
  const Grid g{h.rx, h.ry, h.rz};
  const int N = g.n(), C = h.channels();
  const std::size_t per = zu(N) * kPhys;
  Vec S, phys, pressure(zu(N), 0.f), noise(zu(N) * kNoise), tmp;
  state_of(m, r, first, S, phys);
  const std::vector<float> cond = r.controls();
  Model noise_model;  // the coarse noise needs a workspace without fine fields
  noise_model.h = h;
  noise_model.h.fine = 1;
  noise_model.noise = m.noise;
  noise_model.detail = m.detail;
  noise_model.fps = m.fps;
  Workspace ws(noise_model);
  const auto noise_at = [&](int state) { coarse_noise(noise_model, r.p.seed, r.time(state), ws, noise); };
  int i0 = first;
  if (burn > 0) {  // the stepper's own rollout, no gradient: it learns to correct its own drift
    Cache cb;
    for (int s = 0; s < burn; ++s) {
      noise_at(i0 + s);
      forward(m, L, S.data(), noise.data(), cond, pressure, cb, tmp);
      S = cb.next;
    }
    i0 += burn;
  }
  std::vector<Cache> cs(zu(unroll));
  std::vector<Vec> gout(zu(unroll));
  std::vector<Vec> gprev(zu(unroll), Vec(activity > 0.f ? zu(N) * zu(C) : 0, 0.f));
  std::vector<Vec> outs(zu(unroll));
  std::array<float, kPhys> wch{};
  for (int k = 0; k < kPhys; ++k) wch[zu(k)] = 1.f / (m.scale[zu(k)] * m.scale[zu(k)]);
  const float inv = 1.f / (fl(N) * kPhys * fl(unroll));
  std::mt19937 rng(static_cast<unsigned>(noise_seed));
  std::normal_distribution<float> nd(0.f, 1.f);
  Vec t(per), tp(per);
  double loss = 0;
  for (int s = 0; s < unroll; ++s) {
    noise_at(i0 + s);
    if (sigma > 0.f) {
      for (int j = 0; j < N; ++j) {
        for (int k = 0; k < kPhys; ++k) {
          float& x = S[zu(j) * zu(C) + zu(k)];
          x = std::clamp(x + sigma * m.scale[zu(k)] * nd(rng), m.lo[zu(k)], m.hi[zu(k)]);
        }
      }
    }
    forward(m, L, S.data(), noise.data(), cond, pressure, cs[zu(s)], tmp);
    S = cs[zu(s)].next;
    if (activity > 0.f) outs[zu(s)] = S;
    r.state(i0 + s + 1, per, t);
    Vec& gs = gout[zu(s)];
    gs.assign(zu(N) * zu(C), 0.f);
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < kPhys; ++k) {
        const float e = S[zu(j) * zu(C) + zu(k)] - t[zu(j) * kPhys + zu(k)];
        loss += static_cast<double>(wch[zu(k)] * e * e * inv);
        gs[zu(j) * zu(C) + zu(k)] = 2.f * wch[zu(k)] * e * inv;
      }
    }
    if (profile > 0.f) {  // where the heat and soot are: their means over slices along each axis
      const float pinv = profile / fl(2 * (g.rx + g.ry + g.rz) * unroll);
      for (int k = 3; k < kPhys; ++k) {
        for (int axis = 0; axis < 3; ++axis) {
          const int bins = axis == 0 ? g.rx : (axis == 1 ? g.ry : g.rz);
          const float per_slice = fl(N / bins) * m.scale[zu(k)];
          Vec dm(zu(bins), 0.f);
          for (int y = 0; y < g.ry; ++y) {
            for (int z = 0; z < g.rz; ++z) {
              for (int x = 0; x < g.rx; ++x) {
                const std::size_t j = g.at(x, y, z);
                dm[zu(axis == 0 ? x : (axis == 1 ? y : z))] += (S[j * zu(C) + zu(k)] - t[j * kPhys + zu(k)]) / per_slice;
              }
            }
          }
          for (int b = 0; b < bins; ++b) loss += static_cast<double>(pinv * dm[zu(b)] * dm[zu(b)]);
          for (int y = 0; y < g.ry; ++y) {
            for (int z = 0; z < g.rz; ++z) {
              for (int x = 0; x < g.rx; ++x) {
                const std::size_t j = g.at(x, y, z);
                gs[j * zu(C) + zu(k)] += 2.f * pinv * dm[zu(axis == 0 ? x : (axis == 1 ? y : z))] / per_slice;
              }
            }
          }
        }
      }
    }
    if (activity > 0.f && s > 0) {  // how much each channel changes per frame: mean squared change, model against truth
      r.state(i0 + s, per, tp);
      const Vec& prev = outs[zu(s - 1)];
      for (int k = 0; k < kPhys; ++k) {
        const float is2 = wch[zu(k)];
        double am = 0, at = 0;
        for (int j = 0; j < N; ++j) {
          const float dmod = S[zu(j) * zu(C) + zu(k)] - prev[zu(j) * zu(C) + zu(k)];
          const float dtru = t[zu(j) * kPhys + zu(k)] - tp[zu(j) * kPhys + zu(k)];
          am += static_cast<double>(dmod * dmod * is2);
          at += static_cast<double>(dtru * dtru * is2);
        }
        am /= N;
        at /= N;
        const float e = static_cast<float>(am - at), wgt = activity / (fl(kPhys) * fl(unroll - 1));
        loss += static_cast<double>(wgt * e * e);
        for (int j = 0; j < N; ++j) {
          const float dmod = S[zu(j) * zu(C) + zu(k)] - prev[zu(j) * zu(C) + zu(k)];
          const float gj = wgt * 2.f * e * 2.f * dmod * is2 / fl(N);
          gs[zu(j) * zu(C) + zu(k)] += gj;
          gprev[zu(s - 1)][zu(j) * zu(C) + zu(k)] -= gj;
        }
      }
    }
  }
  if (grad) {
    Vec gS(zu(N) * zu(C), 0.f), gp(zu(N), 0.f), gS_in, gp_in;
    for (int s = unroll - 1; s >= 0; --s) {
      for (std::size_t j = 0; j < gS.size(); ++j) gS[j] += gout[zu(s)][j];
      if (activity > 0.f) {
        for (std::size_t j = 0; j < gS.size(); ++j) gS[j] += gprev[zu(s)][j];
      }
      backward(m, L, cs[zu(s)], cond, gS, gp, gS_in, gp_in, grad->data());
      gS.swap(gS_in);
      gp.swap(gp_in);
    }
  }
  return loss;
}

// --- checkpoints ----------------------------------------------------------------------------------------------------

std::expected<void, std::string> save_checkpoint(const std::filesystem::path& path, const Model& m, const TrainState& s) {
  const std::filesystem::path tmp = path.string() + ".tmp";
  {
    std::ofstream o(tmp, std::ios::binary);
    if (!o) return std::unexpected("cannot write " + tmp.string());
    o.write(kCkptMagic, 8);
    bin::put(o, static_cast<std::int32_t>(s.iteration));
    bin::put(o, static_cast<std::int32_t>(s.adam_t));
    bin::put(o, static_cast<std::uint64_t>(m.step_w.size()));
    bin::put_array(o, std::span<const float>(m.step_w));
    bin::put_array(o, std::span<const float>(s.m1));
    bin::put_array(o, std::span<const float>(s.m2));
    for (const auto* a : {&m.scale, &m.lo, &m.hi}) bin::put_array(o, std::span<const float>(*a));
    bin::put(o, static_cast<std::uint32_t>(s.curve.size()));
    for (const auto& [it, l] : s.curve) {
      bin::put(o, static_cast<std::int32_t>(it));
      bin::put(o, l);
    }
    if (!o) return std::unexpected("write failed: " + tmp.string());
  }
  std::filesystem::rename(tmp, path);
  return {};
}

std::expected<void, std::string> load_checkpoint(const std::filesystem::path& path, Model& m, TrainState& s) {
  std::ifstream i(path, std::ios::binary);
  if (!i) return std::unexpected("cannot open " + path.string());
  char magic[8];
  i.read(magic, 8);
  if (!i || std::memcmp(magic, kCkptMagic, 8) != 0) return std::unexpected("not a 3D checkpoint");
  auto it = bin::get<std::int32_t>(i);
  auto at = bin::get<std::int32_t>(i);
  auto n = bin::get<std::uint64_t>(i);
  if (!it || !at || !n || *n != m.step_w.size()) return std::unexpected("checkpoint does not match the model");
  s.iteration = *it;
  s.adam_t = *at;
  s.m1.resize(*n);
  s.m2.resize(*n);
  for (std::vector<float>* v : {&m.step_w, &s.m1, &s.m2}) {
    if (auto e = bin::get_array(i, std::span<float>(*v)); !e) return std::unexpected(e.error());
  }
  for (auto* a : {&m.scale, &m.lo, &m.hi}) {
    if (auto e = bin::get_array(i, std::span<float>(*a)); !e) return std::unexpected(e.error());
  }
  auto cn = bin::get<std::uint32_t>(i);
  if (!cn) return std::unexpected(cn.error());
  s.curve.clear();
  for (std::uint32_t k = 0; k < *cn; ++k) {
    auto a = bin::get<std::int32_t>(i);
    auto l = bin::get<double>(i);
    if (!a || !l) return std::unexpected("truncated checkpoint");
    s.curve.emplace_back(*a, *l);
  }
  return {};
}

void train_stepper(Model& m, std::span<const Run> runs, const TrainOptions& o, TrainState& s,
                   const std::function<void(const Model&, const TrainState&)>& checkpoint) {
  if (runs.empty()) throw std::invalid_argument("rollout3d: no runs");
  const StepLayout L = step_layout(m.h);
  if (m.step_w.size() != L.size) throw std::invalid_argument("rollout3d: weights do not match the hyperparameters");
  const std::size_t per = zu(m.h.cells()) * kPhys;
  for (const Run& r : runs) {
    if (r.coarse.size() != zu(r.frames) * per || r.frames < o.max_unroll + o.burn_max + 2) throw std::invalid_argument("rollout3d: a run has the wrong shape");
  }
  if (s.m1.size() != L.size) {
    s.m1.assign(L.size, 0.f);
    s.m2.assign(L.size, 0.f);
  }
  Pool pool(o.threads);
  const int total = o.iterations + o.finetune + o.activity_stage;
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<Vec> grads(zu(o.batch), Vec(L.size, 0.f));
  std::vector<double> losses(zu(o.batch), 0.0);
  for (int it = s.iteration; it < total; ++it) {
    const bool fine = it >= o.iterations;
    const bool act = it >= o.iterations + o.finetune;
    const int unroll = fine ? o.max_unroll : std::min(o.max_unroll, 1 + it * o.max_unroll / std::max(1, o.iterations / 2));
    pool.run(o.batch, [&](int b) {
      std::mt19937_64 rng(o.seed * 1000003ULL + static_cast<std::uint64_t>(it) * 977ULL + static_cast<std::uint64_t>(b) * 7919ULL);
      const Run& r = runs[rng() % runs.size()];
      const int burn = fine && o.burn_max > 0 && (b & 1) ? static_cast<int>(rng() % static_cast<std::uint64_t>(o.burn_max + 1)) : 0;
      const int span = r.frames - unroll - burn - 1;
      const int first = static_cast<int>(rng() % static_cast<std::uint64_t>(std::max(1, span)));
      std::ranges::fill(grads[zu(b)], 0.f);
      losses[zu(b)] = window_loss(m, r, first, unroll, burn, o.sigma, fine ? o.profile : 0.f, rng(), &grads[zu(b)], act ? o.activity : 0.f);
    });
    Vec& gsum = grads[0];
    double loss = losses[0];
    for (int b = 1; b < o.batch; ++b) {
      loss += losses[zu(b)];
      for (std::size_t j = 0; j < L.size; ++j) gsum[j] += grads[zu(b)][j];
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
    const int stage_it = act ? it - o.iterations - o.finetune : (fine ? it - o.iterations : it);
    const int stage_n = act ? o.activity_stage : (fine ? o.finetune : o.iterations);
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
    s.iteration = it + 1;
    if (it % std::max(1, o.log_every) == 0 || it == total - 1) {
      s.curve.emplace_back(it, loss);
      if (o.progress) o.progress(it, unroll, loss, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    if (checkpoint && ((o.checkpoint_every > 0 && s.iteration % o.checkpoint_every == 0) || s.iteration == total)) checkpoint(m, s);
  }
}

void choose_starts(Model& m, std::span<const Run> runs, int count, int frame) {
  m.starts.clear();
  if (runs.empty() || count <= 0) return;
  const std::size_t per = zu(m.h.cells()) * kPhys;
  // farthest-point order in control space, from the run nearest the middle
  std::vector<int> chosen;
  const auto dist = [&](const Run& a, const std::vector<float>& c) {
    const auto x = a.controls();
    float d = 0;
    for (std::size_t k = 0; k < x.size(); ++k) d += (x[k] - c[k]) * (x[k] - c[k]);
    return d;
  };
  int first = 0;
  for (int i = 1; i < static_cast<int>(runs.size()); ++i) {
    if (dist(runs[zu(i)], {0.5f, 0.5f, 0.5f}) < dist(runs[zu(first)], {0.5f, 0.5f, 0.5f})) first = i;
  }
  chosen.push_back(first);
  while (static_cast<int>(chosen.size()) < std::min(count, static_cast<int>(runs.size()))) {
    int best = -1;
    float best_d = -1.f;
    for (int i = 0; i < static_cast<int>(runs.size()); ++i) {
      if (std::ranges::find(chosen, i) != chosen.end()) continue;
      float d = 1e30f;
      for (const int c : chosen) d = std::min(d, dist(runs[zu(i)], runs[zu(c)].controls()));
      if (d > best_d) {
        best_d = d;
        best = i;
      }
    }
    chosen.push_back(best);
  }
  for (const int i : chosen) {
    const Run& r = runs[zu(i)];
    StartPoint sp;
    sp.controls = r.controls();
    sp.seed = r.p.seed;
    const int f = std::min(frame, r.frames - 1);
    sp.time = r.time(f);
    sp.coarse.resize(per);
    r.state(f, per, sp.coarse);
    m.starts.push_back(std::move(sp));
  }
}

}  // namespace nfx::rollout3d

#ifdef NFX_ROLLOUT3D_PUSHED
#pragma GCC pop_options
#endif
