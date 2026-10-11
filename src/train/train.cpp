#include <neuralfx/train.hpp>

#include "net.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>
#include <numbers>
#include <numeric>
#include <random>
#include <ranges>
#include <stdexcept>
#include <stdfloat>
#include <thread>

namespace nfx::train {

bool cpu_supported() {
#if defined(__GNUC__) && defined(__x86_64__)
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
  return false;
#endif
}

namespace {

using detail::Grads;
using detail::Net;

// Adam moments for one tensor.
struct Moments {
  std::vector<float> m, v;
  explicit Moments(std::size_t n = 0) : m(n, 0.f), v(n, 0.f) {}
};

struct Adam {
  float b1 = 0.9f, b2 = 0.99f;
  int step = 0;
  void update(std::span<float> p, std::span<const float> g, Moments& s, float lr, float eps, std::size_t begin,
              std::size_t end) const {
    const float c1 = 1.f - std::pow(b1, static_cast<float>(step)), c2 = 1.f - std::pow(b2, static_cast<float>(step));
    for (std::size_t i = begin; i < end; ++i) {
      s.m[i] = b1 * s.m[i] + (1.f - b1) * g[i];
      s.v[i] = b2 * s.v[i] + (1.f - b2) * g[i] * g[i];
      p[i] -= lr * (s.m[i] / c1) / (std::sqrt(s.v[i] / c2) + eps);
    }
  }
  void update(std::span<float> p, std::span<const float> g, Moments& s, float lr, float eps) const {
    update(p, g, s, lr, eps, 0, p.size());
  }
};

void check(const Hyper& h, std::span<const Example> data) {
  if (data.empty()) throw std::invalid_argument("train: no examples");
  const int size = data[0].clip->size;
  for (const Example& e : data) {
    if (!e.clip) throw std::invalid_argument("train: example without a clip");
    if (e.clip->size != size) throw std::invalid_argument("train: clips differ in size");
    if (e.clip->frames != h.frames) throw std::invalid_argument("train: clip length must equal Hyper::frames");
    if (static_cast<int>(e.controls.size()) != h.n_controls) throw std::invalid_argument("train: wrong number of controls");
  }
  if (h.arch == Arch::conv && size != h.size) throw std::invalid_argument("train: conv family must train at its native size");
}

// Vector quantisation during training (Options::vq_bits): the codebook, the latest assignment, and use counts.
struct Vq {
  int G = 0, d = 0, K = 0, C = 0;
  std::size_t S2 = 0, slices = 0;
  std::vector<float> cb;            // [G][K][d]
  std::vector<std::uint16_t> idx;   // [slice][G][S2]
  std::vector<int> idle;            // steps since each codeword was last used

  float* code(int g, int k) { return cb.data() + (static_cast<std::size_t>(g) * K + k) * d; }
  void vec(const Model& m, std::size_t sl, int g, std::size_t j, float* v) const {
    const float* f = m.features.data() + sl * static_cast<std::size_t>(C) * S2;
    for (int c = 0; c < d; ++c) v[c] = f[static_cast<std::size_t>(g * d + c) * S2 + j];
  }
  // Nearest codewords of the model's features into idx.
  void assign(const Model& m) {
    std::vector<float> v(static_cast<std::size_t>(d));
    for (std::size_t sl = 0; sl < slices; ++sl) {
      for (int g = 0; g < G; ++g) {
        const float* base = code(g, 0);
        for (std::size_t j = 0; j < S2; ++j) {
          vec(m, sl, g, j, v.data());
          int best = 0;
          float bd = std::numeric_limits<float>::infinity();
          for (int k = 0; k < K; ++k) {
            const float* w = base + static_cast<std::size_t>(k) * d;
            float dist = 0;
            for (int c = 0; c < d; ++c) dist += (v[static_cast<std::size_t>(c)] - w[c]) * (v[static_cast<std::size_t>(c)] - w[c]);
            if (dist < bd) {
              bd = dist;
              best = k;
            }
          }
          idx[(sl * static_cast<std::size_t>(G) + static_cast<std::size_t>(g)) * S2 + j] = static_cast<std::uint16_t>(best);
        }
      }
    }
  }
  // Features replaced by their assigned codewords.
  void apply(Model& m) {
    for (std::size_t sl = 0; sl < slices; ++sl) {
      float* f = m.features.data() + sl * static_cast<std::size_t>(C) * S2;
      for (int g = 0; g < G; ++g) {
        for (std::size_t j = 0; j < S2; ++j) {
          const float* w = code(g, idx[(sl * static_cast<std::size_t>(G) + static_cast<std::size_t>(g)) * S2 + j]);
          for (int c = 0; c < d; ++c) f[static_cast<std::size_t>(g * d + c) * S2 + j] = w[c];
        }
      }
    }
  }
  // Codewords towards the mean of their assigned features (rate a); unused ones restart at a random feature vector.
  void update(const Model& m, float a, std::mt19937_64& rng) {
    std::vector<double> sum(cb.size(), 0.0);
    std::vector<int> count(static_cast<std::size_t>(G) * K, 0);
    std::vector<float> v(static_cast<std::size_t>(d));
    for (std::size_t sl = 0; sl < slices; ++sl) {
      for (int g = 0; g < G; ++g) {
        for (std::size_t j = 0; j < S2; ++j) {
          const int k = idx[(sl * static_cast<std::size_t>(G) + static_cast<std::size_t>(g)) * S2 + j];
          vec(m, sl, g, j, v.data());
          const std::size_t o = static_cast<std::size_t>(g) * K + static_cast<std::size_t>(k);
          ++count[o];
          for (int c = 0; c < d; ++c) sum[o * static_cast<std::size_t>(d) + static_cast<std::size_t>(c)] += v[static_cast<std::size_t>(c)];
        }
      }
    }
    std::uniform_int_distribution<std::size_t> pick_sl(0, slices - 1), pick_j(0, S2 - 1);
    for (int g = 0; g < G; ++g) {
      for (int k = 0; k < K; ++k) {
        const std::size_t o = static_cast<std::size_t>(g) * K + static_cast<std::size_t>(k);
        float* w = code(g, k);
        if (count[o] > 0) {
          idle[o] = 0;
          for (int c = 0; c < d; ++c) {
            const float mean = static_cast<float>(sum[o * static_cast<std::size_t>(d) + static_cast<std::size_t>(c)] / count[o]);
            w[c] += a * (mean - w[c]);
          }
        } else if (++idle[o] > 50) {
          idle[o] = 0;
          vec(m, pick_sl(rng), g, pick_j(rng), w);
        }
      }
    }
  }
  // k-means from K distinct random feature vectors, 15 Lloyd iterations.
  void init(const Model& m, int bits, int dim, std::mt19937_64& rng) {
    const Hyper& h = m.h;
    C = h.feature_channels();
    d = dim;
    G = C / dim;
    K = 1 << bits;
    S2 = static_cast<std::size_t>(h.feature_side()) * h.feature_side();
    slices = static_cast<std::size_t>(h.bases) * h.grid_t;
    cb.assign(static_cast<std::size_t>(G) * K * d, 0.f);
    idx.assign(slices * static_cast<std::size_t>(G) * S2, 0);
    idle.assign(static_cast<std::size_t>(G) * K, 0);
    std::uniform_int_distribution<std::size_t> pick_sl(0, slices - 1), pick_j(0, S2 - 1);
    for (int g = 0; g < G; ++g) {
      for (int k = 0; k < K; ++k) vec(m, pick_sl(rng), g, pick_j(rng), code(g, k));
    }
    for (int it = 0; it < 15; ++it) {
      assign(m);
      std::vector<int> keep(idle);
      update(m, 1.f, rng);
      idle = keep;
    }
    std::ranges::fill(idle, 0);
  }
};

}  // namespace

namespace {

// The time slices a frame at model time t blends, and the weight of the second (as Net::begin_frame).
void slices_at(const Hyper& h, float t, int& i0, int& i1, float& ft) {
  if (h.loop) {
    const float u = (t - std::floor(t)) * static_cast<float>(h.grid_t);
    i0 = std::min(static_cast<int>(u), h.grid_t - 1);
    i1 = (i0 + 1) % h.grid_t;
    ft = u - static_cast<float>(i0);
  } else {
    const float u = std::clamp(t, 0.f, 1.f) * static_cast<float>(h.grid_t - 1);
    i0 = std::min(static_cast<int>(u), h.grid_t - 2);
    i1 = i0 + 1;
    ft = u - static_cast<float>(i0);
  }
}

}  // namespace

void store_planes(Model& m, std::span<const std::uint8_t> plane_bits, bool trim) {
  const std::size_t S2 = static_cast<std::size_t>(m.h.feature_side()) * m.h.feature_side();
  const std::size_t C = static_cast<std::size_t>(m.h.feature_channels()), T = static_cast<std::size_t>(m.h.grid_t);
  if (!plane_bits.empty() && plane_bits.size() * S2 != m.features.size()) throw std::invalid_argument("store_planes: one width per feature plane");
  if (!m.feature_mask.empty() && m.feature_mask.size() != T * S2) throw std::invalid_argument("store_planes: a mask of grid_t planes");
  for (std::size_t off = 0, k = 0; off < m.features.size(); off += S2, ++k) {
    const auto active = m.feature_mask.empty() ? std::span<const std::uint8_t>{} : std::span<const std::uint8_t>(m.feature_mask).subspan(((k / C) % T) * S2, S2);
    quantise_plane(std::span(m.features.data() + off, S2), plane_bits.empty() ? 16 : plane_bits[k], trim, active);
  }
}

void fake_quantise(Model& m, std::span<const std::uint8_t> plane_bits, bool trim) {
  if (plane_bits.empty()) throw std::invalid_argument("fake_quantise: one width per feature plane");
  store_planes(m, plane_bits, trim);
}

std::vector<std::uint8_t> feature_support(const Hyper& h, std::span<const Example> data, int threshold, int dilate) {
  if (h.arch != Arch::grid || data.empty()) throw std::invalid_argument("feature_support: the grid family, and examples");
  const int G = h.grid, T = h.grid_t, S = data[0].clip->size;
  const std::size_t G2 = static_cast<std::size_t>(G) * G;
  // Per pixel column (and row): the grid points it samples with a weight above zero (-1: none).
  std::vector<std::array<int, 2>> cover(static_cast<std::size_t>(S));
  const float gmax = static_cast<float>(G - 1);
  for (int x = 0; x < S; ++x) {
    const float g = std::clamp((static_cast<float>(x) + 0.5f) / static_cast<float>(S) * static_cast<float>(G) - 0.5f, 0.f, gmax);
    const int x0 = std::min(static_cast<int>(g), G - 2);
    const float fx = g - static_cast<float>(x0);
    cover[static_cast<std::size_t>(x)] = {fx < 1.f ? x0 : -1, fx > 0.f ? x0 + 1 : -1};
  }
  std::vector<std::uint8_t> mask(static_cast<std::size_t>(T) * G2, 0);
  std::vector<std::uint8_t> on(G2);
  for (const Example& e : data) {
    for (int f = 0; f < e.clip->frames && f < h.frames; ++f) {
      std::ranges::fill(on, std::uint8_t{0});
      const auto fr = e.clip->frame(f);
      for (int y = 0; y < S; ++y) {
        for (int x = 0; x < S; ++x) {
          const std::size_t i = (static_cast<std::size_t>(y) * S + x) * 4;
          if (std::max({fr[i], fr[i + 1], fr[i + 2], fr[i + 3]}) <= threshold) continue;
          for (const int gy : cover[static_cast<std::size_t>(y)]) {
            for (const int gx : cover[static_cast<std::size_t>(x)]) {
              if (gy >= 0 && gx >= 0) on[static_cast<std::size_t>(gy) * G + gx] = 1;
            }
          }
        }
      }
      int i0 = 0, i1 = 0;
      float ft = 0;
      slices_at(h, frame_time(h, f, h.frames), i0, i1, ft);
      for (const int sl : {i0, ft > 0.f ? i1 : i0}) {
        std::uint8_t* d = mask.data() + static_cast<std::size_t>(sl) * G2;
        for (std::size_t j = 0; j < G2; ++j) d[j] |= on[j];
      }
    }
  }
  for (int step = 0; step < dilate; ++step) {
    const std::vector<std::uint8_t> before = mask;
    for (int t = 0; t < T; ++t) {
      for (int y = 0; y < G; ++y) {
        for (int x = 0; x < G; ++x) {
          std::uint8_t v = 0;
          for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
              const int yy = y + dy, xx = x + dx;
              if (yy >= 0 && xx >= 0 && yy < G && xx < G) v |= before[static_cast<std::size_t>(t) * G2 + static_cast<std::size_t>(yy) * G + xx];
            }
          }
          mask[static_cast<std::size_t>(t) * G2 + static_cast<std::size_t>(y) * G + x] = v;
        }
      }
    }
  }
  return mask;
}

std::vector<std::uint8_t> allocate_plane_bits(const Model& m, std::span<const Example> data, std::span<const std::vector<float>> codes,
                                              double avg_bits, int min_bits, int max_bits, bool trim, int threads, int size,
                                              std::vector<double>* distortion) {
  if (min_bits < 0 || max_bits > 8 || min_bits > max_bits || data.empty() || avg_bits < min_bits || avg_bits > max_bits) {
    throw std::invalid_argument("allocate_plane_bits: bits 0 to 8, min <= average <= max, and examples");
  }
  const Hyper& h = m.h;
  const std::size_t S2 = static_cast<std::size_t>(h.feature_side()) * h.feature_side();
  const std::size_t C = static_cast<std::size_t>(h.feature_channels()), T = static_cast<std::size_t>(h.grid_t);
  const std::size_t planes = m.features.size() / S2;
  const int rs = h.arch == Arch::conv ? h.size : size > 0 ? size : data[0].clip->size;
  const std::size_t L = static_cast<std::size_t>(max_bits - min_bits + 1);
  const std::size_t px = static_cast<std::size_t>(rs) * rs * 4;
  threads = std::max(1, threads);

  struct Frame {
    std::size_t e;
    float t;
  };
  std::vector<Frame> frames;
  std::vector<std::vector<std::size_t>> by_slice(T);  // the frames that blend each time slice with a weight above zero
  std::vector<std::vector<float>> conds;
  for (std::size_t e = 0; e < data.size(); ++e) {
    const std::vector<float> zero(static_cast<std::size_t>(h.n_latent), 0.f);
    conds.push_back(condition(m, data[e].controls, e < codes.size() && !codes[e].empty() ? std::span<const float>(codes[e]) : std::span<const float>(zero)));
    for (int f = 0; f < h.frames; ++f) {
      const float t = frame_time(h, f, h.frames);
      int i0 = 0, i1 = 0;
      float ft = 0;
      slices_at(h, t, i0, i1, ft);
      by_slice[static_cast<std::size_t>(i0)].push_back(frames.size());
      if (ft > 0.f && i1 != i0) by_slice[static_cast<std::size_t>(i1)].push_back(frames.size());
      frames.push_back({e, t});
    }
  }
  // The frames as the float model renders them (with a mask: the points outside it at their planes' fills).
  Model base = m;
  if (!base.feature_mask.empty()) store_planes(base, {}, trim);
  std::vector<float> ref(frames.size() * px);
  const auto run = [&](auto&& body) {
    std::vector<std::jthread> pool;
    for (int t = 0; t < threads; ++t) pool.emplace_back(body, t);
  };
  run([&](int t) {
    Net net(base);
    for (std::size_t i = static_cast<std::size_t>(t); i < frames.size(); i += static_cast<std::size_t>(threads)) {
      net.render(base, frames[i].t, conds[frames[i].e], rs, std::span(ref.data() + i * px, px));
    }
  });
  // Each plane alone at each width: the squared change of the frames that use it.
  std::vector<double> D(planes * L, 0.0);
  std::atomic<std::size_t> next{0};
  run([&](int) {
    Model mt = base;
    Net net(mt);
    std::vector<float> out(px), keep(S2);
    for (std::size_t P = next++; P < planes; P = next++) {
      const std::size_t slice = (P / C) % T;  // planes are [basis][slice][channel]
      const auto active = mt.feature_mask.empty() ? std::span<const std::uint8_t>{} : std::span<const std::uint8_t>(mt.feature_mask).subspan(slice * S2, S2);
      float* pl = mt.features.data() + P * S2;
      std::copy_n(pl, S2, keep.begin());
      for (int b = min_bits; b <= max_bits; ++b) {
        quantise_plane(std::span(pl, S2), b, trim, active);
        double d = 0;
        for (const std::size_t i : by_slice[slice]) {
          net.render(mt, frames[i].t, conds[frames[i].e], rs, out);
          const float* r = ref.data() + i * px;
          for (std::size_t j = 0; j < px; ++j) d += static_cast<double>(out[j] - r[j]) * static_cast<double>(out[j] - r[j]);
        }
        D[P * L + static_cast<std::size_t>(b - min_bits)] = d;
        std::copy_n(keep.begin(), S2, pl);
      }
    }
  });
  // Bits to the steepest fall in distortion per bit, along each plane's lower convex hull.
  std::vector<std::uint8_t> bits(planes, static_cast<std::uint8_t>(min_bits));
  long budget = std::lround(avg_bits * static_cast<double>(planes)) - static_cast<long>(min_bits) * static_cast<long>(planes);
  while (budget > 0) {
    double best = 0;
    std::size_t best_p = planes;
    int best_j = 0;
    for (std::size_t P = 0; P < planes; ++P) {
      const int b = bits[P];
      for (int j = 1; b + j <= max_bits && j <= budget; ++j) {
        const double slope = (D[P * L + static_cast<std::size_t>(b - min_bits)] - D[P * L + static_cast<std::size_t>(b + j - min_bits)]) / j;
        if (slope > best) {
          best = slope;
          best_p = P;
          best_j = j;
        }
      }
    }
    if (best_p == planes) break;  // no plane gains from another bit
    bits[best_p] = static_cast<std::uint8_t>(bits[best_p] + best_j);
    budget -= best_j;
  }
  if (distortion) *distortion = std::move(D);
  return bits;
}

void fake_quantise(Model& m, int bits, bool trim) {
  // Only the features: quantise_like_storage also rounds the other weights to fp16, which training must not do.
  const std::size_t plane = static_cast<std::size_t>(m.h.feature_side()) * m.h.feature_side();
  const float qm = static_cast<float>((1 << bits) - 1);
  for (std::size_t off = 0; off < m.features.size(); off += plane) {
    const std::span p(m.features.data() + off, plane);
    const auto [lo, hi] = feature_plane_range(p, bits, trim);
    for (float& v : p) {
      const float q = std::clamp(static_cast<float>(std::lround((v - lo) / (hi - lo) * qm)), 0.f, qm);
      v = lo + q / qm * (hi - lo);
    }
  }
}

double feature_rate(const Model& m, int bits_all, std::span<float> grad, float weight, bool trim, std::span<const std::uint8_t> plane_bits) {
  const Hyper& h = m.h;
  const int S = h.feature_side(), C = h.feature_channels(), T = h.grid_t;
  const std::size_t plane = static_cast<std::size_t>(S) * S;
  const float inv_ln2 = 1.f / std::numbers::ln2_v<float>;
  double total = 0;
  for (int k = 0; k < h.bases; ++k) {
    for (int t = 0; t < T; ++t) {
      for (int c = 0; c < C; ++c) {
        const std::size_t index = (static_cast<std::size_t>(k) * T + t) * C + c;
        const int bits = plane_bits.empty() ? bits_all : plane_bits[index];
        if (bits == 0) continue;  // one value per plane: nothing per value to code
        const float qm = static_cast<float>((1 << bits) - 1);
        const std::size_t off = index * plane;
        const float* p = m.features.data() + off;
        const float* q = t > 0 ? p - static_cast<std::size_t>(C) * plane : nullptr;  // previous time slice, same channel
        const auto [plo, phi] = feature_plane_range(std::span(p, plane), bits, trim);
        const float step = std::max(phi - plo, 1e-6f) / qm;
        float* g = grad.empty() ? nullptr : grad.data() + off;
        float* gq = g && q ? g - static_cast<std::size_t>(C) * plane : nullptr;
        for (int y = 0; y < S; ++y) {
          for (int x = 0; x < S; ++x) {
            const std::size_t i = static_cast<std::size_t>(y) * S + x;
            const float v = p[i];
            // candidate predictions: value and the indices (and signs) it is made of
            float best = std::numeric_limits<float>::infinity(), res = 0;
            int which = -1;
            std::size_t a = 0, b = 0, cc = 0;  // MED pieces
            int med_kind = 0;                  // 0: W only, 1: N only, 2: min/max picks one, 3: W + N - NW
            if (x > 0 || y > 0) {
              float pred;
              if (x > 0 && y > 0) {
                const float W = p[i - 1], N = p[i - S], NW = p[i - S - 1];
                const float lo = std::min(W, N), hi = std::max(W, N);
                if (NW >= hi) {
                  pred = lo;
                  a = W <= N ? i - 1 : i - S;
                  med_kind = 2;
                } else if (NW <= lo) {
                  pred = hi;
                  a = W >= N ? i - 1 : i - S;
                  med_kind = 2;
                } else {
                  pred = W + N - NW;
                  a = i - 1;
                  b = i - S;
                  cc = i - S - 1;
                  med_kind = 3;
                }
              } else if (x > 0) {
                pred = p[i - 1];
                a = i - 1;
                med_kind = 2;
              } else {
                pred = p[i - S];
                a = i - S;
                med_kind = 2;
              }
              const float r = v - pred;
              if (std::abs(r) < best) {
                best = std::abs(r);
                res = r;
                which = 0;
              }
            }
            if (q) {
              const float r1 = v - q[i];
              if (std::abs(r1) < best) {
                best = std::abs(r1);
                res = r1;
                which = 1;
              }
              if (x > 0) {
                const float r2 = v - (q[i] + p[i - 1] - q[i - 1]);
                if (std::abs(r2) < best) {
                  best = std::abs(r2);
                  res = r2;
                  which = 2;
                }
              }
            }
            if (which < 0) continue;  // the first value of the first slice: nothing to predict it from
            const float u = best / step;
            total += std::log2(1.0 + static_cast<double>(u));
            if (!g) continue;
            // d log2(1 + |r| / step) / dr, then r = v - prediction
            const float d = weight * (res > 0 ? 1.f : -1.f) * inv_ln2 / (step * (1.f + u));
            g[i] += d;
            if (which == 0) {
              if (med_kind == 2) {
                g[a] -= d;
              } else {
                g[a] -= d;
                g[b] -= d;
                g[cc] += d;
              }
            } else if (which == 1) {
              gq[i] -= d;
            } else {
              gq[i] -= d;
              g[i - 1] -= d;
              gq[i - 1] += d;
            }
          }
        }
      }
    }
  }
  return total;
}

Result train(const Hyper& h_in, std::span<const Example> data, const Options& o) {
  if (!cpu_supported()) throw std::runtime_error("train: this CPU lacks AVX2 + FMA");
  Hyper h = o.init ? o.init->h : h_in;
  check(h, data);
  const auto t_start = std::chrono::steady_clock::now();
  Model m = o.init ? *o.init : init_model(h, o.seed);
  const int size = data[0].clip->size;
  const int Z = h.n_latent, NC = h.n_controls;
  std::mt19937_64 rng(o.seed * 0x9e3779b97f4a7c15ULL + 7);

  // Variation codes: reuse the init model's if they match, else small random values.
  std::vector<std::vector<float>> codes(data.size(), std::vector<float>(static_cast<std::size_t>(Z), 0.f));
  if (o.init && o.init->z_train.size() == data.size()) {
    codes = o.init->z_train;
  } else {
    std::normal_distribution<float> nz(0.f, 0.01f);
    for (auto& z : codes) std::ranges::generate(z, [&] { return nz(rng); });
  }

  const int threads = std::clamp(o.threads > 0 ? o.threads : static_cast<int>(std::thread::hardware_concurrency()), 1,
                                 std::max(1, o.batch_frames));
  std::vector<Net> nets;
  std::vector<Grads> grads;
  for (int t = 0; t < threads; ++t) {
    nets.emplace_back(m);
    grads.emplace_back(m);
  }
  Grads total(m);
  std::vector<int> frames = o.frames;
  if (frames.empty()) frames = std::views::iota(0, h.frames) | std::ranges::to<std::vector>();

  // Optimiser state, one Moments per tensor in a fixed order.
  Moments mf(m.features.size()), mbw(m.basis.w.size()), mbb(m.basis.b.size());
  std::vector<Moments> mlw, mlb, mfw, mfb;
  for (const auto& d : m.layers) {
    mlw.emplace_back(d.w.size());
    mlb.emplace_back(d.b.size());
  }
  for (const auto& d : m.films) {
    mfw.emplace_back(d.w.size());
    mfb.emplace_back(d.b.size());
  }
  std::vector<Moments> mz(codes.size(), Moments(static_cast<std::size_t>(Z)));
  Adam adam;

  const int pixels_per_frame = h.arch == Arch::grid ? std::min(o.pixels, size * size) : size * size;
  const float scale = 1.f / (static_cast<float>(o.batch_frames) * static_cast<float>(pixels_per_frame) * 4.f);
  const std::size_t plane = static_cast<std::size_t>(h.feature_channels()) * h.feature_side() * h.feature_side();
  Result res;
  std::vector<double> losses;
  losses.reserve(static_cast<std::size_t>(o.iterations));
  std::uniform_int_distribution<std::size_t> pick_ex(0, data.size() - 1), pick_fr(0, frames.size() - 1);

  if (o.qat_bits != 0 && (o.qat_bits < 2 || o.qat_bits > 8)) throw std::invalid_argument("train: qat_bits must be 0 or 2 to 8");
  if (o.vq_bits != 0 && (o.vq_bits < 2 || o.vq_bits > 8 || o.vq_dim < 1 || h.feature_channels() % o.vq_dim != 0 || o.qat_bits != 0)) {
    throw std::invalid_argument("train: vq_bits 2 to 8 and vq_dim dividing the channels, without qat_bits");
  }
  const int vq_from = o.vq_bits > 0 ? static_cast<int>(std::lround(static_cast<double>(o.vq_start) * o.iterations)) : o.iterations + 1;
  Vq vq;
  std::mt19937_64 vq_rng(o.seed * 7919 + 3);
  if (o.rate_lambda > 0.f && (o.rate_bits < 2 || o.rate_bits > 8)) throw std::invalid_argument("train: rate_bits must be 2 to 8");
  const bool mixed = o.mixed_bits > 0.f;
  if (mixed && (o.qat_bits != 0 || o.vq_bits != 0 || o.mixed_min < 0 || o.mixed_max > 8 || o.mixed_min > o.mixed_max ||
                o.mixed_bits < static_cast<float>(o.mixed_min) || o.mixed_bits > static_cast<float>(o.mixed_max))) {
    throw std::invalid_argument("train: mixed_bits between mixed_min and mixed_max (0 to 8), without qat_bits or vq_bits");
  }
  std::vector<std::uint8_t> plane_bits;  // per plane: chosen when quantisation starts (mixed), or all qat_bits (sparse)
  if (o.sparse) {
    if (h.arch != Arch::grid || o.vq_bits != 0 || o.rate_lambda > 0.f) {
      throw std::invalid_argument("train: sparse features need the grid family, without vq_bits or a rate term");
    }
    m.feature_mask = feature_support(h, data, o.sparse_threshold, o.sparse_dilate);
    if (o.qat_bits > 0) plane_bits.assign(m.features.size() / (static_cast<std::size_t>(h.grid) * h.grid), static_cast<std::uint8_t>(o.qat_bits));
  }
  const bool per_plane = mixed || o.sparse;
  const int qat_from = static_cast<int>(std::lround(static_cast<double>(o.qat_start) * o.iterations));
  Model fwd;  // the model the forward pass sees: features as stored (quantisation-aware training)
  double rate = 0;

  const auto allocate = [&] {
    const auto t0 = std::chrono::steady_clock::now();
    plane_bits = allocate_plane_bits(m, data, codes, o.mixed_bits, o.mixed_min, o.mixed_max, o.qat_trim, threads, o.mixed_size);
    res.alloc_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  };
  for (int it = 0; it < o.iterations; ++it) {
    const bool qat = (o.qat_bits > 0 || mixed) && it >= qat_from;
    const bool vq_on = it >= vq_from;
    if (it == vq_from) {
      vq.init(m, o.vq_bits, o.vq_dim, vq_rng);
      vq.assign(m);
    }
    if (mixed && it == qat_from) allocate();
    if (per_plane && (qat || o.sparse)) {
      fwd = m;
      store_planes(fwd, qat ? std::span<const std::uint8_t>(plane_bits) : std::span<const std::uint8_t>{}, o.qat_trim);
    } else if (qat) {
      fwd = m;
      fake_quantise(fwd, o.qat_bits, o.qat_trim);
    } else if (vq_on) {
      fwd = m;
      vq.apply(fwd);
    }
    const Model& seen = qat || vq_on || o.sparse ? fwd : m;
    // The minibatch: (example, frame) pairs, dealt round-robin to the threads.
    std::vector<std::pair<std::size_t, int>> batch(static_cast<std::size_t>(o.batch_frames));
    for (auto& b : batch) b = {pick_ex(rng), frames[pick_fr(rng)]};
    std::vector<double> sse(static_cast<std::size_t>(threads), 0.0);
    std::vector<std::vector<float>> dcode(static_cast<std::size_t>(threads),
                                          std::vector<float>(codes.size() * static_cast<std::size_t>(Z), 0.f));
    const std::uint64_t it_seed = rng();
    const auto work = [&](int t) {
      grads[static_cast<std::size_t>(t)].zero();
      std::mt19937_64 prng(it_seed + static_cast<std::uint64_t>(t) * 0x632be59bd9b4e019ULL);
      std::uniform_int_distribution<int> pick_px(0, size * size - 1);
      std::vector<int> px(static_cast<std::size_t>(h.arch == Arch::grid ? pixels_per_frame : 0));
      std::vector<float> dc(static_cast<std::size_t>(h.dims()));
      for (std::size_t b = static_cast<std::size_t>(t); b < batch.size(); b += static_cast<std::size_t>(threads)) {
        const auto [e, f] = batch[b];
        const auto c = condition(m, data[e].controls, codes[e]);
        for (int& p : px) p = pick_px(prng);
        std::ranges::sort(px);  // better locality in the feature planes
        std::ranges::fill(dc, 0.f);
        sse[static_cast<std::size_t>(t)] += nets[static_cast<std::size_t>(t)].step(
            seen, grads[static_cast<std::size_t>(t)], frame_time(h, f, h.frames), c, data[e].clip->frame(f), px, size, scale, dc);
        for (int j = 0; j < Z; ++j) dcode[static_cast<std::size_t>(t)][e * static_cast<std::size_t>(Z) + static_cast<std::size_t>(j)] += dc[static_cast<std::size_t>(NC + j)];
      }
    };
    if (threads == 1) {
      work(0);
    } else {
      std::vector<std::jthread> pool;
      for (int t = 0; t < threads; ++t) pool.emplace_back(work, t);
    }
    total.zero();
    for (const auto& g : grads) total.add(g);
    const double loss = std::ranges::fold_left(sse, 0.0, std::plus{}) / (static_cast<double>(o.batch_frames) * pixels_per_frame * 4.0);
    losses.push_back(loss);
    if (o.rate_lambda > 0.f && !o.freeze_model) {  // the rate term reaches every slice
      rate = feature_rate(m, o.rate_bits, total.g.features, o.rate_lambda / static_cast<float>(m.features.size()), o.qat_trim, plane_bits) /
             static_cast<double>(m.features.size());
      std::ranges::fill(total.touched, std::uint8_t{1});
    }

    // Cosine decay of every learning rate to final_lr_scale.
    const float progress = static_cast<float>(it) / static_cast<float>(std::max(1, o.iterations - 1));
    const float k = o.final_lr_scale + (1.f - o.final_lr_scale) * 0.5f * (1.f + std::cos(std::numbers::pi_v<float> * progress));
    ++adam.step;
    if (!o.freeze_model) {
      // Features: only the time slices this step touched (sparse Adam).
      for (int s = 0; s < h.grid_t; ++s) {
        if (!total.touched[static_cast<std::size_t>(s)]) continue;
        for (int b = 0; b < h.bases; ++b) {
          const std::size_t off = (static_cast<std::size_t>(b) * h.grid_t + s) * plane;
          adam.update(m.features, total.g.features, mf, o.lr_features * k, 1e-10f, off, off + plane);
        }
      }
      adam.update(m.basis.w, total.g.basis.w, mbw, o.lr * k, 1e-8f);
      adam.update(m.basis.b, total.g.basis.b, mbb, o.lr * k, 1e-8f);
      for (std::size_t l = 0; l < m.layers.size(); ++l) {
        adam.update(m.layers[l].w, total.g.layers[l].w, mlw[l], o.lr * k, 1e-8f);
        adam.update(m.layers[l].b, total.g.layers[l].b, mlb[l], o.lr * k, 1e-8f);
      }
      for (std::size_t l = 0; l < m.films.size(); ++l) {
        adam.update(m.films[l].w, total.g.films[l].w, mfw[l], o.lr * k, 1e-8f);
        adam.update(m.films[l].b, total.g.films[l].b, mfb[l], o.lr * k, 1e-8f);
      }
    }
    if (Z > 0) {
      // Codes: data term plus the |z|^2 prior (mean over examples).
      std::vector<float> g(static_cast<std::size_t>(Z));
      for (std::size_t e = 0; e < codes.size(); ++e) {
        bool used = false;
        for (int j = 0; j < Z; ++j) {
          float d = 0;
          for (int t = 0; t < threads; ++t) d += dcode[static_cast<std::size_t>(t)][e * static_cast<std::size_t>(Z) + static_cast<std::size_t>(j)];
          used = used || d != 0.f;
          g[static_cast<std::size_t>(j)] = d + 2.f * o.z_prior * codes[e][static_cast<std::size_t>(j)] / static_cast<float>(codes.size());
        }
        if (used || o.z_prior > 0.f) adam.update(codes[e], g, mz[e], o.lr_codes * k, 1e-8f);
      }
    }
    if (vq_on && !o.freeze_model) {  // the next assignment, and the codebook towards it
      vq.assign(m);
      vq.update(m, 0.2f, vq_rng);
    }
    if (o.log_every > 0 && (it % o.log_every == 0 || it + 1 == o.iterations)) {
      res.curve.emplace_back(it, loss);
      if (o.progress) o.progress(it, loss);
    }
  }

  if (mixed && plane_bits.empty()) allocate();  // quantisation never started: bits for the trained model
  res.final_rate = o.rate_lambda > 0.f ? rate
                                        : feature_rate(m, o.qat_bits > 0 ? o.qat_bits : 8, {}, 0.f, o.qat_trim, plane_bits) /
                                              static_cast<double>(m.features.size());
  const std::size_t tail = std::max<std::size_t>(1, losses.size() / 20);
  res.final_loss = std::accumulate(losses.end() - static_cast<std::ptrdiff_t>(tail), losses.end(), 0.0) / static_cast<double>(tail);
  if (Z > 0) m.z_train = codes;  // no codes to keep without a variation dimension
  else m.z_train.clear();
  m.z_mean.assign(static_cast<std::size_t>(Z), 0.f);
  m.z_std.assign(static_cast<std::size_t>(Z), 0.f);
  for (const auto& z : codes) {
    for (int j = 0; j < Z; ++j) m.z_mean[static_cast<std::size_t>(j)] += z[static_cast<std::size_t>(j)] / static_cast<float>(codes.size());
  }
  for (const auto& z : codes) {
    for (int j = 0; j < Z; ++j) {
      const float d = z[static_cast<std::size_t>(j)] - m.z_mean[static_cast<std::size_t>(j)];
      m.z_std[static_cast<std::size_t>(j)] += d * d / static_cast<float>(codes.size());
    }
  }
  for (float& s : m.z_std) s = std::sqrt(s);
  if (o.vq_bits > 0 && vq_from <= o.iterations) {
    if (vq.cb.empty()) vq.init(m, o.vq_bits, o.vq_dim, vq_rng);  // vq_start = 1: a codebook fitted after training
    m.vq_bits = o.vq_bits;
    m.vq_dim = o.vq_dim;
    m.vq_codebook = vq.cb;
  }
  if (per_plane) m.plane_bits = std::move(plane_bits);  // (sparse without quantisation: no widths; the caller sets them)
  res.model = std::move(m);
  res.codes = std::move(codes);
  res.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
  return res;
}

Clip render_clip(const Model& m, std::span<const float> controls, std::span<const float> z, int frames, int size) {
  if (!cpu_supported()) throw std::runtime_error("render_clip: this CPU lacks AVX2 + FMA");
  Clip clip;
  clip.allocate(size, frames);
  clip.loop = m.h.loop;
  clip.effect = m.effect;
  clip.source = "model";
  clip.fps = m.fps;
  clip.n_controls = static_cast<int>(std::min<std::size_t>(controls.size(), kMaxControls));
  for (int i = 0; i < clip.n_controls; ++i) clip.controls[static_cast<std::size_t>(i)] = controls[static_cast<std::size_t>(i)];
  const auto c = condition(m, controls, z);
  const int threads = std::clamp(static_cast<int>(std::thread::hardware_concurrency()), 1, frames);
  {
    std::vector<std::jthread> pool;
    for (int t = 0; t < threads; ++t) {
      pool.emplace_back([&, t] {
        Net net(m);
        std::vector<float> rgba(static_cast<std::size_t>(size) * size * 4);
        for (int f = t; f < frames; f += threads) {
          net.render(m, frame_time(m.h, f, frames), c, size, rgba);
          to_rgba8(rgba, clip.frame(f));
        }
      });
    }
  }
  return clip;
}

}  // namespace nfx::train
