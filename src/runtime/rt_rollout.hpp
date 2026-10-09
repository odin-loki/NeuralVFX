// Rollout effects in the runtime: included by rt_impl.hpp inside the per-ISA namespace (after the target pragma), so
// the loops below are compiled once per ISA. The operations are those of the reference (src/core/rollout.cpp); what
// differs is bookkeeping:
//   - FiLM is folded into the stepper's convolution weights once per frame;
//   - the procedural noise on fixed points (coarse cells, swirl lattice) is value noise, which is a blend of two 2D
//     slices in its third coordinate; the slices are cached and recomputed only when time crosses a lattice step;
//   - the renderer runs on blocks of 16 pixels with the shared dense kernel.
// Every buffer is allocated in the constructor.

using rollout::kDirSteps;
using rollout::kDirs;
using rollout::kNoise;
using rollout::kPhys;
using rollout::kRenderIn;

// Value noise at fixed 2D points, as a function of the third coordinate z. value_noise(x, y, z) is
// lerp(slice(floor z), slice(floor z + 1), smooth5(frac z)); the two slices are kept per point.
class SliceNoise {
 public:
  void init(std::size_t points, std::uint64_t seed) {
    seed_ = seed;
    x_.assign(points, 0.f);
    y_.assign(points, 0.f);
    a_.assign(points, 0.f);
    b_.assign(points, 0.f);
    iz_ = std::numeric_limits<std::int64_t>::min();
  }
  void set(std::size_t i, float x, float y) {
    x_[i] = x;
    y_[i] = y;
    iz_ = std::numeric_limits<std::int64_t>::min();
  }
  // Refresh the slices if z moved to another lattice step.
  void at(float z) {
    const float fz = std::floor(z);
    const auto iz = static_cast<std::int64_t>(fz);
    tz_ = smooth5(z - fz);
    if (iz == iz_) return;
    if (iz == iz_ + 1) {
      a_.swap(b_);
      fill(b_, static_cast<std::int32_t>(iz + 1));
    } else {
      fill(a_, static_cast<std::int32_t>(iz));
      fill(b_, static_cast<std::int32_t>(iz + 1));
    }
    iz_ = iz;
  }
  float value(std::size_t i) const { return a_[i] + tz_ * (b_[i] - a_[i]); }
  std::size_t bytes() const { return 4 * (x_.size() + y_.size() + a_.size() + b_.size()); }

 private:
  void fill(std::vector<float>& s, std::int32_t iz) const {
    for (std::size_t i = 0; i < s.size(); ++i) {
      const float fx = std::floor(x_[i]), fy = std::floor(y_[i]);
      const auto ix = static_cast<std::int32_t>(fx), iy = static_cast<std::int32_t>(fy);
      const float tx = smooth5(x_[i] - fx), ty = smooth5(y_[i] - fy);
      const float c00 = cell_value(ix, iy, iz, seed_), c10 = cell_value(ix + 1, iy, iz, seed_);
      const float c01 = cell_value(ix, iy + 1, iz, seed_), c11 = cell_value(ix + 1, iy + 1, iz, seed_);
      const float a = c00 + tx * (c10 - c00), b = c01 + tx * (c11 - c01);
      s[i] = a + ty * (b - a);
    }
  }
  std::uint64_t seed_ = 0;
  std::vector<float> x_, y_, a_, b_;
  std::int64_t iz_ = 0;
  float tz_ = 0.f;
};

// Fractal value noise (fbm of value_noise) at arbitrary 2D points in a fixed rectangle, as a function of time: each
// octave's lattice corner values for the two time slices around z are cached and refreshed when z crosses a lattice
// step, so a sample is interpolation only. The same corners and interpolation as value_noise, hence the same values.
class LatticeFbm {
 public:
  void init(float freq, float rate, int octaves, std::uint64_t seed, float lo, float hi) {
    freq_ = freq;
    rate_ = rate;
    oct_.resize(static_cast<std::size_t>(octaves));
    float f = freq;
    for (int o = 0; o < octaves; ++o) {
      Octave& q = oct_[static_cast<std::size_t>(o)];
      q.seed = seed + static_cast<std::uint64_t>(o) * 0x9e3779b97f4a7c15ULL;
      q.i0 = static_cast<int>(std::floor(lo * f)) - 1;
      q.n = static_cast<int>(std::floor(hi * f)) - q.i0 + 3;
      q.a.assign(static_cast<std::size_t>(q.n) * static_cast<std::size_t>(q.n), 0.f);
      q.b.assign(q.a.size(), 0.f);
      q.iz = std::numeric_limits<std::int64_t>::min();
      f *= 2.f;
    }
  }
  void at(float t) {
    float z = t * rate_;
    for (Octave& q : oct_) {
      const float fz = std::floor(z);
      const auto iz = static_cast<std::int64_t>(fz);
      q.tz = smooth5(z - fz);
      if (iz != q.iz) {
        if (iz == q.iz + 1) {
          q.a.swap(q.b);
          fill(q, q.b, static_cast<std::int32_t>(iz + 1));
        } else {
          fill(q, q.a, static_cast<std::int32_t>(iz));
          fill(q, q.b, static_cast<std::int32_t>(iz + 1));
        }
        q.iz = iz;
      }
      z *= 2.f;
    }
  }
  float value(float X, float Y) const {
    float sum = 0.f, amp = 1.f, norm = 0.f, x = X * freq_, y = Y * freq_;
    for (const Octave& q : oct_) {
      const float fx = std::floor(x), fy = std::floor(y);
      const int ix = static_cast<int>(fx) - q.i0, iy = static_cast<int>(fy) - q.i0;
      const float tx = smooth5(x - fx), ty = smooth5(y - fy);
      const std::size_t k = static_cast<std::size_t>(iy) * static_cast<std::size_t>(q.n) + static_cast<std::size_t>(ix), n = static_cast<std::size_t>(q.n);
      const auto lerp = [](float a, float b, float t) { return a + t * (b - a); };
      const float a0 = lerp(lerp(q.a[k], q.a[k + 1], tx), lerp(q.a[k + n], q.a[k + n + 1], tx), ty);
      const float a1 = lerp(lerp(q.b[k], q.b[k + 1], tx), lerp(q.b[k + n], q.b[k + n + 1], tx), ty);
      sum += amp * lerp(a0, a1, q.tz);
      norm += amp;
      amp *= 0.5f;
      x *= 2.f;
      y *= 2.f;
    }
    return sum / norm;
  }
  std::size_t bytes() const {
    std::size_t n = 0;
    for (const Octave& q : oct_) n += 4 * (q.a.size() + q.b.size());
    return n;
  }

 private:
  struct Octave {
    std::uint64_t seed = 0;
    int i0 = 0, n = 0;
    std::vector<float> a, b;
    std::int64_t iz = 0;
    float tz = 0.f;
  };
  static void fill(const Octave& q, std::vector<float>& s, std::int32_t iz) {
    for (int j = 0; j < q.n; ++j) {
      for (int i = 0; i < q.n; ++i) s[static_cast<std::size_t>(j) * static_cast<std::size_t>(q.n) + static_cast<std::size_t>(i)] = cell_value(q.i0 + i, q.i0 + j, iz, q.seed);
    }
  }
  float freq_ = 1.f, rate_ = 1.f;
  std::vector<Octave> oct_;
};

class Rollout final : public RolloutRunner {
 public:
  Rollout(const RolloutEffect& e, int size) : m_(e.m), h_(e.m.h), S_(size) {
    using namespace rollout;
    R_ = h_.res;
    N_ = R_ * R_;
    C_ = h_.channels();
    I_ = h_.inputs();
    H_ = h_.hidden;
    O_ = h_.outputs();
    if (S_ % R_ != 0 || S_ < R_) throw std::invalid_argument("rollout: size must be a multiple of the coarse grid");
    const StepLayout L = step_layout(h_);
    L_ = L;
    w1_.resize(9 * z(I_) * z(H_));
    b1_.resize(z(H_));
    w2_.resize(9 * z(H_) * z(H_));
    b2_.resize(z(H_));
    cond_.assign(z(h_.cond()), 0.f);
    folded_cond_.assign(z(h_.cond()), std::numeric_limits<float>::quiet_NaN());
    X_.assign(z(R_ + 2) * z(R_ + 2) * z(I_), 0.f);  // planar, zero border (written inside only)
    h1p_.assign(z(R_ + 2) * z(R_ + 2) * z(H_), 0.f);
    h1_.resize(z(N_) * z(H_));
    h2_.resize(z(N_) * z(H_));
    d_.resize(z(N_) * z(O_));
    mid_.resize(z(N_) * z(C_));
    next_.resize(z(N_) * z(C_));
    coarse_.resize(z(N_) * z(C_));
    flow_.resize(z(N_) * 2);
    div_.resize(z(N_));
    p_.resize(z(N_));
    tmp_.resize(z(N_));
    noise_.resize(z(N_) * kNoise);
    dirsum_.resize(z(N_) * kDirs);
    B_.resize(z(N_));
    rr_.resize(z(N_));
    aa_.resize(z(N_));
    const std::size_t S2 = z(S_) * z(S_);
    ft_.resize(S2);
    fd_.resize(S2);
    ux_.resize(S2);
    vy_.resize(S2);
    const std::size_t P2 = z(S_ + 3) * z(S_ + 3);
    fa_.assign(P2, 0.f);  // padded: zero border, interior written each step
    ga_.assign(P2, 0.f);
    tp_.assign(P2, 0.f);
    dp_.assign(P2, 0.f);
    fb_.resize(S2);
    gb_.resize(S2);
    lo_t_.resize(S2);
    hi_t_.resize(S2);
    lo_d_.resize(S2);
    hi_d_.resize(S2);
    feat_.resize(kRenderIn * kB);
    r1_.resize(z(h_.render_hidden) * kB);
    r2_.resize(z(h_.render_hidden) * kB);
    out_.resize(4 * kB);
    row_.resize(4 * z(S_) * kB);
    // Render weights split into the dense kernel's [out][in] blocks.
    const RenderLayout RL = render_layout(h_);
    RL_ = RL;
    // Noise on fixed points: coarse cell centres (curl stream function; flicker octaves), the swirl lattice.
    const float k = 128.f / static_cast<float>(R_);
    curl_.init(z(N_), 0);
    flicker_.resize(z(m_.noise.flicker_octaves));
    for (auto& f : flicker_) f.init(z(N_), 0);
    sw_spacing_ = 0.5f * m_.detail.swirl_scale;
    sw_n_ = static_cast<int>(std::ceil(130.f / sw_spacing_)) + 3;
    swirl_.init(z(sw_n_) * z(sw_n_), 0);
    swu_.resize(z(sw_n_) * z(sw_n_));
    swv_.resize(z(sw_n_) * z(sw_n_));
    ax_ = axis(S_, R_, static_cast<float>(R_) / static_cast<float>(S_), -0.5f);
    rowa_.resize(z(std::max(2 * R_, sw_n_)));
    rowb_.resize(rowa_.size());
    rowc_.resize(rowa_.size());
    rowd_.resize(rowa_.size());
    B2_.resize(z(N_));
    crow_.resize(z(R_) * (2 + kDirs));
    frow_.resize(z(kRenderIn) * z(S_));
    rr2_.resize(z(N_));
    aa2_.resize(z(N_));
    sx_ = axis(S_, sw_n_, 128.f / static_cast<float>(S_) / sw_spacing_, 0.5f / sw_spacing_ + 1.f);
    reseed(0);  // sizes every noise cache now, so a later begin() with another seed allocates nothing
    (void)k;
  }

  void start(int index, std::span<const float> controls, std::uint64_t seed) override {
    begin(index, seed);
    if (!m_.starts[z(index)].fine_t.empty()) return;
    for (int f = 0; f < h_.warmup; ++f) step(controls, seed);
    since_start_ = m_.detail.swirl_ramp;
  }

  void begin(int index, std::uint64_t seed) override {
    using namespace rollout;
    const StartPoint& sp = m_.starts[z(index)];
    reseed(seed);
    time_ = sp.time;
    std::fill(coarse_.begin(), coarse_.end(), 0.f);
    for (int i = 0; i < N_; ++i) {
      for (int c = 0; c < kPhys; ++c) coarse_[z(i) * z(C_) + z(c)] = sp.coarse[z(i) * kPhys + z(c)];
    }
    std::fill(p_.begin(), p_.end(), 0.f);
    std::fill(flow_.begin(), flow_.end(), 0.f);
    const bool fine = !sp.fine_t.empty();
    const int n = fine ? h_.start_fine : R_;
    const float k = static_cast<float>(S_) / static_cast<float>(n);
    for (int y = 0; y < S_; ++y) {
      for (int x = 0; x < S_; ++x) {
        const float xs = (static_cast<float>(x) + 0.5f) / k - 0.5f, ys = (static_cast<float>(y) + 0.5f) / k - 0.5f;
        const std::size_t i = z(y) * z(S_) + z(x);
        ft_[i] = fine ? bilinear(sp.fine_t.data(), n, 1, 0, xs, ys) : bilinear(sp.coarse.data(), R_, kPhys, 2, xs, ys);
        fd_[i] = fine ? bilinear(sp.fine_d.data(), n, 1, 0, xs, ys) : bilinear(sp.coarse.data(), R_, kPhys, 3, xs, ys);
      }
    }
    since_start_ = fine ? 0.f : m_.detail.swirl_ramp;
  }

  void step(std::span<const float> controls, std::uint64_t seed) override {
    reseed(seed);
    rollout::condition(m_, controls, time_, cond_);
    fold();
    coarse_noise(time_ + 0.5f / m_.fps);
    coarse_step();
    detail_step(controls);
    time_ += 1.f / m_.fps;
    since_start_ += 1.f / m_.fps;
  }

  void render(const FrameInput& in, std::uint8_t* rgba, std::size_t stride) override {
    using namespace rollout;
    // directional soot sums on the coarse grid
    static constexpr int dirs[kDirs][2] = {{1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}, {0, -1}, {1, -1}};
    for (int y = 0; y < R_; ++y) {
      for (int x = 0; x < R_; ++x) {
        for (int j = 0; j < kDirs; ++j) {
          float s = 0.f;
          for (int st = 1; st <= kDirSteps; ++st) {
            const int xx = x + st * dirs[j][0], yy = y + st * dirs[j][1];
            if (xx >= 0 && yy >= 0 && xx < R_ && yy < R_) s += coarse_[(z(yy) * z(R_) + z(xx)) * z(C_) + 3];
          }
          dirsum_[(z(y) * z(R_) + z(x)) * kDirs + z(j)] = s;
        }
      }
    }
    const int RH = h_.render_hidden;
    const float* w = m_.render_w.data();
    const float it = 1.f / m_.render_scale[0], id = 1.f / m_.render_scale[1];
    constexpr int kc = 2 + kDirs;  // coarse features: heat, soot, directional sums
    const std::size_t S = z(S_);
    for (int y = 0; y < S_; ++y) {
      // one row of the coarse features, interpolated in y and normalised
      const float wy = ax_.w[z(y)];
      const std::size_t r0 = z(ax_.i[z(y)]) * z(R_), r1 = r0 + z(R_);
      for (int i = 0; i < R_; ++i) {
        float* o = crow_.data() + z(i) * kc;
        const std::size_t a = r0 + z(i), b = r1 + z(i);
        o[0] = (coarse_[a * z(C_) + 2] + wy * (coarse_[b * z(C_) + 2] - coarse_[a * z(C_) + 2])) * it;
        o[1] = (coarse_[a * z(C_) + 3] + wy * (coarse_[b * z(C_) + 3] - coarse_[a * z(C_) + 3])) * id;
        for (int j = 0; j < kDirs; ++j) o[2 + j] = (dirsum_[a * kDirs + z(j)] + wy * (dirsum_[b * kDirs + z(j)] - dirsum_[a * kDirs + z(j)])) * id;
      }
      // the row's features as full-width planes [feature][S]: the dense kernel reads blocks of them in place
      float* F = frow_.data();
      const float* tf = ft_.data() + z(y) * S;
      const float* df = fd_.data() + z(y) * S;
      for (std::size_t x = 0; x < S; ++x) {
        F[x] = tf[x] * it;
        F[S + x] = df[x] * id;
      }
      for (int j = 0; j < kc; ++j) {
        float* row = F + z(2 + j) * S;
        for (std::size_t x = 0; x < S; ++x) {
          const float* ca = crow_.data() + z(ax_.i[x]) * kc + z(j);
          row[x] = ca[0] + ax_.w[x] * (ca[kc] - ca[0]);
        }
      }
      std::uint8_t* out_row = rgba + stride * z(S_ - 1 - y);
      for (int x0 = 0; x0 < S_; x0 += kB) {  // S_ is a multiple of 32, so blocks of 16 are whole
        dense(w + RL_.w1, w + RL_.b1, F + x0, S_, r1_.data(), kB, kRenderIn, RH, true);
        dense(w + RL_.w2, w + RL_.b2, r1_.data(), kB, r2_.data(), kB, RH, RH, true);
        dense(w + RL_.wo, w + RL_.bo, r2_.data(), kB, out_.data(), kB, RH, 4, false);
        for (int q = 0; q < kB; ++q) {
          const float g = rollout::render_gate(F[z(x0 + q)], F[S + z(x0 + q)]);
          for (int ch = 0; ch < 4; ++ch) out_[z(ch) * kB + z(q)] *= g;
        }
        write_pixels(out_.data(), out_.data() + kB, out_.data() + 2 * kB, out_.data() + 3 * kB, kB, in, out_row + 4 * z(x0));
      }
    }
  }

  std::size_t scratch_bytes() const override {
    std::size_t n = 0;
    for (const auto* v : {&w1_, &b1_, &w2_, &b2_, &cond_, &folded_cond_, &X_, &h1p_, &h1_, &h2_, &d_, &mid_, &next_, &coarse_, &flow_, &div_, &p_,
                          &tmp_, &noise_, &dirsum_, &B_, &rr_, &aa_, &ft_, &fd_, &ux_, &vy_, &fa_, &fb_, &ga_, &gb_, &tp_, &dp_, &lo_t_, &hi_t_, &lo_d_, &hi_d_, &rowa_, &rowb_, &rowc_, &rowd_, &B2_, &rr2_, &aa2_, &crow_, &frow_, &feat_, &r1_, &r2_, &out_,
                          &row_, &swu_, &swv_}) {
      n += v->size() * 4;
    }
    n += curl_.bytes() + swirl_.bytes() + fine_flicker_.bytes();
    for (const auto& f : flicker_) n += f.bytes();
    return n;
  }

  double macs_per_pixel() const override {
    const double coarse = static_cast<double>(N_) * (9.0 * I_ * H_ + 9.0 * H_ * H_ + static_cast<double>(H_) * O_);
    const double renderer = kRenderIn * h_.render_hidden + h_.render_hidden * h_.render_hidden + 4.0 * h_.render_hidden;
    return coarse / (static_cast<double>(S_) * S_) + renderer;
  }

  float time() const override { return time_; }
  std::span<const float> coarse() const override { return coarse_; }
  std::span<const float> fine_heat() const override { return ft_; }
  std::span<const float> fine_soot() const override { return fd_; }
  int size() const override { return S_; }
  std::span<const float> flow() const override { return flow_; }
  std::span<float> coarse_mut() override { return coarse_; }
  std::span<float> fine_heat_mut() override { return ft_; }
  std::span<float> fine_soot_mut() override { return fd_; }
  void adopt(float seconds) override {
    time_ = seconds;
    since_start_ = m_.detail.swirl_ramp;
    std::fill(p_.begin(), p_.end(), 0.f);
    std::fill(flow_.begin(), flow_.end(), 0.f);
  }

 private:
  static std::size_t z(int v) { return static_cast<std::size_t>(v); }

  // Bilinear at cell-centre coordinates on an n x n grid of `channels` interleaved values, clamped.
  static float bilinear(const float* f, int n, int channels, int c, float x, float y) {
    x = std::clamp(x, 0.f, static_cast<float>(n - 1));
    y = std::clamp(y, 0.f, static_cast<float>(n - 1));
    const int x0 = std::min(static_cast<int>(x), n - 2), y0 = std::min(static_cast<int>(y), n - 2);
    const float fx = x - static_cast<float>(x0), fy = y - static_cast<float>(y0);
    const auto at = [&](int xx, int yy) { return f[(z(yy) * z(n) + z(xx)) * z(channels) + z(c)]; };
    return (1.f - fy) * ((1.f - fx) * at(x0, y0) + fx * at(x0 + 1, y0)) + fy * ((1.f - fx) * at(x0, y0 + 1) + fx * at(x0 + 1, y0 + 1));
  }

  // New seed: the noise caches belong to a seed (the positions do not change).
  void reseed(std::uint64_t seed) {
    if (seeded_ && seed == seed_) return;
    seeded_ = true;
    seed_ = seed;
    const rollout::NoiseSpec& ns = m_.noise;
    const float k = 128.f / static_cast<float>(R_);
    curl_.init(z(N_), seed * 0x2545F4914F6CDD1DULL + 77);
    for (int o = 0; o < ns.flicker_octaves; ++o) flicker_[z(o)].init(z(N_), seed + static_cast<std::uint64_t>(o) * 0x9e3779b97f4a7c15ULL);
    for (int y = 0; y < R_; ++y) {
      for (int x = 0; x < R_; ++x) {
        const float X = (static_cast<float>(x) + 0.5f) * k + 0.5f, Y = (static_cast<float>(y) + 0.5f) * k + 0.5f;
        const std::size_t i = z(y) * z(R_) + z(x);
        curl_.set(i, X * ns.curl_scale, Y * ns.curl_scale);
        float sx = X * ns.flicker_freq, sy = Y * ns.flicker_freq;
        for (int o = 0; o < ns.flicker_octaves; ++o) {
          flicker_[z(o)].set(i, sx, sy);
          sx *= 2.f;
          sy *= 2.f;
        }
      }
    }
    fine_flicker_.init(ns.flicker_freq, ns.flicker_rate, ns.flicker_octaves, seed, 0.f, 130.f);
    swirl_.init(z(sw_n_) * z(sw_n_), seed * 0x9E3779B97F4A7C15ULL + 5);
    for (int j = 0; j < sw_n_; ++j) {
      for (int i = 0; i < sw_n_; ++i) swirl_.set(z(j) * z(sw_n_) + z(i), static_cast<float>(i - 1) * 0.5f, static_cast<float>(j - 1) * 0.5f);
    }
  }

  void coarse_noise(float t) {
    const rollout::NoiseSpec& ns = m_.noise;
    curl_.at(t * ns.curl_rate);
    float zt = t * ns.flicker_rate;
    for (int o = 0; o < ns.flicker_octaves; ++o) {
      flicker_[z(o)].at(zt);
      zt *= 2.f;
    }
    for (int i = 0; i < N_; ++i) {
      noise_[z(i) * 2] = curl_.value(z(i));
      float sum = 0.f, amp = 1.f, norm = 0.f;
      for (int o = 0; o < ns.flicker_octaves; ++o) {
        sum += amp * flicker_[z(o)].value(z(i));
        norm += amp;
        amp *= 0.5f;
      }
      noise_[z(i) * 2 + 1] = sum / norm;
    }
  }

  // FiLM folded into the convolution weights: gamma * (W x + b) + beta = (gamma W) x + (gamma b + beta).
  void fold() {
    if (std::equal(cond_.begin(), cond_.end(), folded_cond_.begin())) return;
    const float* w = m_.step_w.data();
    const auto one = [&](std::size_t W, std::size_t Bb, std::size_t G, std::size_t E, int ci, std::vector<float>& wo, std::vector<float>& bo) {
      for (int j = 0; j < H_; ++j) {
        float g = 1.f, e = 0.f;
        for (int k = 0; k < h_.cond(); ++k) {
          g += w[G + z(k) * z(H_) + z(j)] * cond_[z(k)];
          e += w[E + z(k) * z(H_) + z(j)] * cond_[z(k)];
        }
        // stored [tap][in][out]; the planar kernel wants [out][in][tap]
        for (int tap = 0; tap < 9; ++tap) {
          for (int c = 0; c < ci; ++c) wo[(z(j) * z(ci) + z(c)) * 9 + z(tap)] = g * w[W + (z(tap) * z(ci) + z(c)) * z(H_) + z(j)];
        }
        bo[z(j)] = g * w[Bb + z(j)] + e;
      }
    };
    one(L_.w1, L_.b1, L_.g1, L_.e1, I_, w1_, b1_);
    one(L_.w2, L_.b2, L_.g2, L_.e2, H_, w2_, b2_);
    std::copy(cond_.begin(), cond_.end(), folded_cond_.begin());
  }

  void coarse_step() {
    using namespace rollout;
    const float* w = m_.step_w.data();
    // inputs, planar with a zero border, for the shared 3x3 kernel: [input][R + 2][R + 2]
    const int Pw = R_ + 2;
    for (int y = 0; y < R_; ++y) {
      for (int x = 0; x < R_; ++x) {
        const std::size_t i = z(y) * z(R_) + z(x), q = z(y + 1) * z(Pw) + z(x + 1), plane = z(Pw) * z(Pw);
        for (int k = 0; k < C_; ++k) X_[z(k) * plane + q] = coarse_[i * z(C_) + z(k)] / (k < kPhys ? m_.scale[z(k)] : 1.f);
        X_[z(C_) * plane + q] = noise_[i * 2];
        X_[z(C_ + 1) * plane + q] = noise_[i * 2 + 1];
        X_[z(C_ + 2) * plane + q] = (static_cast<float>(x) + 0.5f) / static_cast<float>(R_) * 2.f - 1.f;
        X_[z(C_ + 3) * plane + q] = (static_cast<float>(y) + 0.5f) / static_cast<float>(R_) * 2.f - 1.f;
      }
    }
    conv3x3(X_.data(), w1_.data(), b1_.data(), h1_.data(), I_, H_, R_, R_, true);
    for (int c = 0; c < H_; ++c) {  // into a zero-bordered buffer for the second convolution
      for (int y = 0; y < R_; ++y) {
        std::copy_n(h1_.data() + (z(c) * z(R_) + z(y)) * z(R_), R_, h1p_.data() + (z(c) * z(Pw) + z(y + 1)) * z(Pw) + 1);
      }
    }
    conv3x3(h1p_.data(), w2_.data(), b2_.data(), h2_.data(), H_, H_, R_, R_, true);
    // the 1x1 output layer, planar: d[k][cell]
    for (int k = 0; k < O_; ++k) {
      float* __restrict o = d_.data() + z(k) * z(N_);
      std::fill_n(o, N_, w[L_.bo + z(k)]);
      for (int j = 0; j < H_; ++j) {
        const float wk = w[L_.wo + z(j) * z(O_) + z(k)];
        const float* __restrict a = h2_.data() + z(j) * z(N_);
        for (int i = 0; i < N_; ++i) o[i] += wk * a[i];
      }
    }
    for (int i = 0; i < N_; ++i) {
      for (int k = 0; k < kPhys; ++k) mid_[z(i) * z(C_) + z(k)] = coarse_[z(i) * z(C_) + z(k)] + m_.scale[z(k)] * d_[z(k) * z(N_) + z(i)];
      for (int k = kPhys; k < C_; ++k) mid_[z(i) * z(C_) + z(k)] = std::tanh(coarse_[z(i) * z(C_) + z(k)] + d_[z(k) * z(N_) + z(i)]);
    }
    const auto U = [&](int x, int y, int c) { return mid_[(z(std::clamp(y, 0, R_ - 1)) * z(R_) + z(std::clamp(x, 0, R_ - 1))) * z(C_) + z(c)]; };
    for (int y = 0; y < R_; ++y) {
      for (int x = 0; x < R_; ++x) {
        div_[z(y) * z(R_) + z(x)] = -0.5f * (U(x + 1, y, 0) - U(x - 1, y, 0) + U(x, y + 1, 1) - U(x, y - 1, 1)) + m_.qscale * d_[z(C_) * z(N_) + z(y) * z(R_) + z(x)];
      }
    }
    for (int it = 0; it < h_.jacobi; ++it) {
      for (int y = 0; y < R_; ++y) {
        const float* up = y + 1 < R_ ? p_.data() + z(y + 1) * z(R_) : nullptr;
        const float* dn = y > 0 ? p_.data() + z(y - 1) * z(R_) : nullptr;
        const float* row = p_.data() + z(y) * z(R_);
        float* t = tmp_.data() + z(y) * z(R_);
        const float* dv = div_.data() + z(y) * z(R_);
        for (int x = 0; x < R_; ++x) {
          const float s = (x > 0 ? row[x - 1] : 0.f) + (x + 1 < R_ ? row[x + 1] : 0.f) + (dn ? dn[x] : 0.f) + (up ? up[x] : 0.f);
          t[x] = 0.25f * (dv[x] + s);
        }
      }
      p_.swap(tmp_);
    }
    const auto P = [&](int x, int y) { return (x < 0 || y < 0 || x >= R_ || y >= R_) ? 0.f : p_[z(y) * z(R_) + z(x)]; };
    for (int y = 0; y < R_; ++y) {
      for (int x = 0; x < R_; ++x) {
        const std::size_t i = z(y) * z(R_) + z(x);
        mid_[i * z(C_)] -= 0.5f * (P(x + 1, y) - P(x - 1, y));
        mid_[i * z(C_) + 1] -= 0.5f * (P(x, y + 1) - P(x, y - 1));
        flow_[i * 2] = mid_[i * z(C_)];
        flow_[i * 2 + 1] = mid_[i * z(C_) + 1];
      }
    }
    for (int y = 0; y < R_; ++y) {
      for (int x = 0; x < R_; ++x) {
        const std::size_t i = z(y) * z(R_) + z(x);
        const float px = std::clamp(static_cast<float>(x) - mid_[i * z(C_)], 0.f, static_cast<float>(R_ - 1));
        const float py = std::clamp(static_cast<float>(y) - mid_[i * z(C_) + 1], 0.f, static_cast<float>(R_ - 1));
        const int x0 = std::min(static_cast<int>(px), R_ - 2), y0 = std::min(static_cast<int>(py), R_ - 2);
        const float fx = px - static_cast<float>(x0), fy = py - static_cast<float>(y0);
        const float* a = mid_.data() + (z(y0) * z(R_) + z(x0)) * z(C_);
        const float* b = a + C_;
        const float* c = a + z(R_) * z(C_);
        const float* d = c + C_;
        float* o = next_.data() + i * z(C_);
        for (int k = 0; k < C_; ++k) {
          float v = (1.f - fy) * ((1.f - fx) * a[k] + fx * b[k]) + fy * ((1.f - fx) * c[k] + fx * d[k]);
          if (k < kPhys) v = std::clamp(v, m_.lo[z(k)], m_.hi[z(k)]);
          o[k] = v;
        }
      }
    }
    coarse_.swap(next_);
  }

  static float smoothstep01(float t) {
    t = std::clamp(t, 0.f, 1.f);
    return t * t * (3.f - 2.f * t);
  }

  // Bilinear weights from the coarse grid (or the swirl lattice) to fine pixel centres: they depend on x or y alone.
  struct Axis {
    std::vector<int> i;
    std::vector<float> w;
  };
  static Axis axis(int S, int n, float scale, float offset) {  // coordinate = (p + 0.5) * scale + offset, clamped
    Axis a;
    a.i.resize(z(S));
    a.w.resize(z(S));
    for (int p = 0; p < S; ++p) {
      const float c = std::clamp((static_cast<float>(p) + 0.5f) * scale + offset, 0.f, static_cast<float>(n - 1));
      const int i0 = std::min(static_cast<int>(c), n - 2);
      a.i[z(p)] = i0;
      a.w[z(p)] = c - static_cast<float>(i0);
    }
    return a;
  }

  // f (n x n, `channels` interleaved, channel c) at fine pixel (x, y) through precomputed axes.
  static float up(const float* f, int n, int channels, int c, const Axis& ax, const Axis& ay, int x, int y) {
    const std::size_t i = (z(ay.i[z(y)]) * z(n) + z(ax.i[z(x)])) * z(channels) + z(c);
    const float fx = ax.w[z(x)], fy = ay.w[z(y)];
    const float a = f[i] + fx * (f[i + z(channels)] - f[i]);
    const float b = f[i + z(n) * z(channels)] + fx * (f[i + z(n + 1) * z(channels)] - f[i + z(n) * z(channels)]);
    return a + fy * (b - a);
  }

  void detail_step(std::span<const float> controls) {
    using namespace rollout;
    const DetailSpec& dt = m_.detail;
    const float k = static_cast<float>(S_) / static_cast<float>(R_), px128 = static_cast<float>(S_) / 128.f, t = time_ + 0.5f / m_.fps;
    float amp = dt.swirl * px128;
    if (dt.swirl_control >= 0 && z(dt.swirl_control) < controls.size()) amp *= 0.3f + controls[z(dt.swirl_control)];
    if (dt.swirl_ramp > 0.f) amp *= std::min(1.f, since_start_ / dt.swirl_ramp);
    if (amp > 0.f) {
      swirl_.at(t * dt.swirl_rate);
      const int n = sw_n_;
      std::fill(swu_.begin(), swu_.end(), 0.f);
      std::fill(swv_.begin(), swv_.end(), 0.f);
      for (int j = 1; j < n - 1; ++j) {
        for (int i = 1; i < n - 1; ++i) {
          const std::size_t q = z(j) * z(n) + z(i);
          swu_[q] = swirl_.value(q + z(n)) - swirl_.value(q - z(n));
          swv_[q] = swirl_.value(q - 1) - swirl_.value(q + 1);
        }
      }
    }
    // fine velocity: the coarse flow (and the swirl), interpolated separably (a row of the grid, then along x)
    for (int y = 0; y < S_; ++y) {
      const float wy = ax_.w[z(y)];
      const float* r0 = flow_.data() + z(ax_.i[z(y)]) * z(R_) * 2;
      const float* r1 = r0 + z(R_) * 2;
      for (int i = 0; i < 2 * R_; ++i) rowa_[z(i)] = (r0[i] + wy * (r1[i] - r0[i])) * k;
      float* uo = ux_.data() + z(y) * z(S_);
      float* vo = vy_.data() + z(y) * z(S_);
      for (int x = 0; x < S_; ++x) {
        const std::size_t i = z(ax_.i[z(x)]) * 2;
        const float wx = ax_.w[z(x)];
        uo[x] = rowa_[i] + wx * (rowa_[i + 2] - rowa_[i]);
        vo[x] = rowa_[i + 1] + wx * (rowa_[i + 3] - rowa_[i + 1]);
      }
      if (amp > 0.f) {
        const float sy = sx_.w[z(y)];
        const float* su0 = swu_.data() + z(sx_.i[z(y)]) * z(sw_n_);
        const float* sv0 = swv_.data() + z(sx_.i[z(y)]) * z(sw_n_);
        for (int i = 0; i < sw_n_; ++i) {
          rowb_[z(i)] = amp * (su0[i] + sy * (su0[i + sw_n_] - su0[i]));
          rowc_[z(i)] = amp * (sv0[i] + sy * (sv0[i + sw_n_] - sv0[i]));
        }
        for (int x = 0; x < S_; ++x) {
          const std::size_t i = z(sx_.i[z(x)]);
          const float wx = sx_.w[z(x)];
          uo[x] += rowb_[i] + wx * (rowb_[i + 1] - rowb_[i]);
          vo[x] += rowc_[i] + wx * (rowc_[i + 1] - rowc_[i]);
        }
      }
    }
    // MacCormack for heat and soot together, zero outside the frame. The fields are copied into buffers with a zero
    // border, so a backtrace clamped to [-1, size] reads zeros outside without a branch: the forward samples and the
    // clamp range in one pass, the round trip in a second, the correction in a third.
    const int Pw = S_ + 3;  // one zero column and row before, two after (a clamped backtrace at the far edge reads two zeros)
    for (int y = 0; y < S_; ++y) {
      std::copy_n(ft_.data() + z(y) * z(S_), S_, tp_.data() + z(y + 1) * z(Pw) + 1);
      std::copy_n(fd_.data() + z(y) * z(S_), S_, dp_.data() + z(y + 1) * z(Pw) + 1);
    }
    const float hi_lim = static_cast<float>(S_);
    const auto backtrace = [&](float px, float py, std::size_t& a, float& fx, float& fy) {
      px = std::clamp(px, -1.f, hi_lim) + 1.f;  // padded coordinates in [0, size + 1]
      py = std::clamp(py, -1.f, hi_lim) + 1.f;
      const int x0 = static_cast<int>(px), y0 = static_cast<int>(py);  // up to size + 1
      fx = px - static_cast<float>(x0);
      fy = py - static_cast<float>(y0);
      a = z(y0) * z(Pw) + z(x0);
    };
    for (int y = 0; y < S_; ++y) {
      for (int x = 0; x < S_; ++x) {
        const std::size_t i = z(y) * z(S_) + z(x);
        std::size_t a;
        float fx, fy;
        backtrace(static_cast<float>(x) - ux_[i], static_cast<float>(y) - vy_[i], a, fx, fy);
        const std::size_t c = a + z(Pw);
        const float t00 = tp_[a], t10 = tp_[a + 1], t01 = tp_[c], t11 = tp_[c + 1];
        const float d00 = dp_[a], d10 = dp_[a + 1], d01 = dp_[c], d11 = dp_[c + 1];
        const float ta = t00 + fx * (t10 - t00), tb = t01 + fx * (t11 - t01);
        const float da = d00 + fx * (d10 - d00), db = d01 + fx * (d11 - d01);
        const std::size_t o = z(y + 1) * z(Pw) + z(x + 1);
        fa_[o] = ta + fy * (tb - ta);
        ga_[o] = da + fy * (db - da);
        lo_t_[i] = std::min(std::min(t00, t10), std::min(t01, t11));
        hi_t_[i] = std::max(std::max(t00, t10), std::max(t01, t11));
        lo_d_[i] = std::min(std::min(d00, d10), std::min(d01, d11));
        hi_d_[i] = std::max(std::max(d00, d10), std::max(d01, d11));
      }
    }
    for (int y = 0; y < S_; ++y) {
      for (int x = 0; x < S_; ++x) {
        const std::size_t i = z(y) * z(S_) + z(x);
        std::size_t a;
        float fx, fy;
        backtrace(static_cast<float>(x) + ux_[i], static_cast<float>(y) + vy_[i], a, fx, fy);
        const std::size_t c = a + z(Pw);
        const float ta = fa_[a] + fx * (fa_[a + 1] - fa_[a]), tb = fa_[c] + fx * (fa_[c + 1] - fa_[c]);
        const float da = ga_[a] + fx * (ga_[a + 1] - ga_[a]), db = ga_[c] + fx * (ga_[c + 1] - ga_[c]);
        fb_[i] = ta + fy * (tb - ta);
        gb_[i] = da + fy * (db - da);
      }
    }
    for (int y = 0; y < S_; ++y) {
      const std::size_t o = z(y + 1) * z(Pw) + 1;
      for (int x = 0; x < S_; ++x) {
        const std::size_t i = z(y) * z(S_) + z(x);
        ft_[i] = std::max(0.f, std::clamp(fa_[o + z(x)] + 0.5f * (ft_[i] - fb_[i]), lo_t_[i], hi_t_[i]));
        fd_[i] = std::max(0.f, std::clamp(ga_[o + z(x)] + 0.5f * (fd_[i] - gb_[i]), lo_d_[i], hi_d_[i]));
      }
    }
    // lock to the coarse state (as the reference), heat and soot together so the flicker is evaluated once per pixel
    const int kk = S_ / R_;
    const float inv = 1.f / static_cast<float>(kk * kk);
    std::fill(B_.begin(), B_.end(), 0.f);
    std::fill(B2_.begin(), B2_.end(), 0.f);
    for (int y = 0; y < S_; ++y) {
      float* bt = B_.data() + z(y / kk) * z(R_);
      float* bd = B2_.data() + z(y / kk) * z(R_);
      const float* qt = ft_.data() + z(y) * z(S_);
      const float* qd = fd_.data() + z(y) * z(S_);
      for (int x = 0; x < S_; ++x) {
        bt[x / kk] += qt[x];
        bd[x / kk] += qd[x];
      }
    }
    constexpr float eps = 1e-4f;
    for (int i = 0; i < N_; ++i) {
      const float bt = B_[z(i)] * inv, tt = coarse_[z(i) * z(C_) + 2];
      const float bd = B2_[z(i)] * inv, td = coarse_[z(i) * z(C_) + 3];
      const float rt = (tt + eps) / (bt + eps), rd = (td + eps) / (bd + eps);
      rr_[z(i)] = rt <= 1.f ? rt : std::min(rt, dt.grow);
      aa_[z(i)] = std::max(0.f, tt - bt * rr_[z(i)]);
      rr2_[z(i)] = rd <= 1.f ? rd : std::min(rd, dt.grow);
      aa2_[z(i)] = std::max(0.f, td - bd * rr2_[z(i)]);
    }
    const float span = 1.f / (dt.edge1 - dt.edge0);
    if (dt.contrast > 0.f) fine_flicker_.at(t);
    for (int y = 0; y < S_; ++y) {
      const float wy = ax_.w[z(y)];
      const std::size_t r0 = z(ax_.i[z(y)]) * z(R_), r1 = r0 + z(R_);
      for (int i = 0; i < R_; ++i) {
        rowa_[z(i)] = aa_[r0 + z(i)] + wy * (aa_[r1 + z(i)] - aa_[r0 + z(i)]);
        rowb_[z(i)] = rr_[r0 + z(i)] + wy * (rr_[r1 + z(i)] - rr_[r0 + z(i)]);
        rowc_[z(i)] = aa2_[r0 + z(i)] + wy * (aa2_[r1 + z(i)] - aa2_[r0 + z(i)]);
        rowd_[z(i)] = rr2_[r0 + z(i)] + wy * (rr2_[r1 + z(i)] - rr2_[r0 + z(i)]);
      }
      float* qt = ft_.data() + z(y) * z(S_);
      float* qd = fd_.data() + z(y) * z(S_);
      const float Y = (static_cast<float>(y) + 0.5f) / px128 + 0.5f;
      for (int x = 0; x < S_; ++x) {
        const std::size_t i = z(ax_.i[z(x)]);
        const float wx = ax_.w[z(x)];
        float at = rowa_[i] + wx * (rowa_[i + 1] - rowa_[i]);
        float ad = rowc_[i] + wx * (rowc_[i + 1] - rowc_[i]);
        if ((at > 0.f || ad > 0.f) && dt.contrast > 0.f) {
          const float X = (static_cast<float>(x) + 0.5f) / px128 + 0.5f;
          const float phi = fine_flicker_.value(X, Y);
          const float g = (1.f - dt.contrast) + dt.contrast * dt.kappa * smoothstep01((phi - dt.edge0) * span);
          at *= g;
          ad *= g;
        }
        qt[x] = qt[x] * (rowb_[i] + wx * (rowb_[i + 1] - rowb_[i])) + at;
        qd[x] = qd[x] * (rowd_[i] + wx * (rowd_[i + 1] - rowd_[i])) + ad;
      }
    }
  }

  const rollout::Model& m_;
  const rollout::Hyper& h_;
  int S_, R_ = 0, N_ = 0, C_ = 0, I_ = 0, H_ = 0, O_ = 0;
  rollout::StepLayout L_{};
  rollout::RenderLayout RL_{};
  std::vector<float> w1_, b1_, w2_, b2_, cond_, folded_cond_;
  std::vector<float> X_, h1p_, h1_, h2_, d_, mid_, next_, coarse_, flow_, div_, p_, tmp_, noise_, dirsum_, B_, rr_, aa_;
  std::vector<float> ft_, fd_, ux_, vy_, fa_, fb_, ga_, gb_, tp_, dp_, lo_t_, hi_t_, lo_d_, hi_d_, feat_, r1_, r2_, out_, row_, swu_, swv_;
  Axis ax_, sx_;  // fine pixel -> coarse cell, fine pixel -> swirl lattice
  std::vector<float> rowa_, rowb_, rowc_, rowd_;  // one interpolated row of a coarse field or of the swirl lattice
  std::vector<float> B2_, rr2_, aa2_;             // the lock's block sums and factors for soot (B_, rr_, aa_: heat)
  std::vector<float> crow_;                       // one row of the renderer's coarse features
  std::vector<float> frow_;                       // one row of all renderer features, planar [feature][size]
  SliceNoise curl_, swirl_;
  LatticeFbm fine_flicker_;
  std::vector<SliceNoise> flicker_;
  float sw_spacing_ = 1.f;
  int sw_n_ = 0;
  std::uint64_t seed_ = 0;
  bool seeded_ = false;
  float time_ = 0.f, since_start_ = 0.f;
};
