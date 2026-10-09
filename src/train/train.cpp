#include <neuralfx/train.hpp>

#include "net.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numbers>
#include <numeric>
#include <random>
#include <ranges>
#include <stdexcept>
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

}  // namespace

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

  for (int it = 0; it < o.iterations; ++it) {
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
            m, grads[static_cast<std::size_t>(t)], frame_time(h, f, h.frames), c, data[e].clip->frame(f), px, size, scale, dc);
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
    if (o.log_every > 0 && (it % o.log_every == 0 || it + 1 == o.iterations)) {
      res.curve.emplace_back(it, loss);
      if (o.progress) o.progress(it, loss);
    }
  }

  const std::size_t tail = std::max<std::size_t>(1, losses.size() / 20);
  res.final_loss = std::accumulate(losses.end() - static_cast<std::ptrdiff_t>(tail), losses.end(), 0.0) / static_cast<double>(tail);
  m.z_train = codes;
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
