// nvfx_swirl: study L (docs/REPORT.md §12, "Study L"): why fast or regular swirling flows fail as rollout effects, and
// whether a fix beats the current models of magic (a spinning portal), the orb (a plasma ball) and the vortex (a dust
// devil).
//
//   nvfx_swirl diag   --effect E --model F     where the tracking error comes from, on training runs (salt 1): the
//                                              learned renderer, the stepper, the detail layer (on the true coarse
//                                              states: an oracle stepper), the coarse simulation, with detail variants
//   nvfx_swirl dstats --effect E --model F     the same split for the endless statistics, on training settings
//   nvfx_swirl column --effect vortex --model F   where the vortex loses its column (training settings)
//
// Options: --root DATA (default $NEURALVFX_DATA or /root/nvfx-data; this study's files are under DATA/l_swirl),
// --results DIR (results/experiments/swirl), --threads N (1).
#include "args.hpp"

#include <neuralfx/clip.hpp>
#include <neuralfx/dcm/mixer.hpp>
#include <neuralfx/image_io.hpp>
#include <neuralfx/metrics.hpp>
#include <neuralfx/nvfx.h>
#include <neuralfx/rollout.hpp>
#include <neuralfx/rollout_train.hpp>
#include <neuralfx/sim.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <mutex>
#include <numeric>
#include <print>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using namespace nfx;

namespace {

constexpr int kSize = 128, kRes = 32;
std::size_t sz(int v) { return static_cast<std::size_t>(v); }
float fl(int v) { return static_cast<float>(v); }
std::string ename(sim::Effect e) { return std::string(sim::effect_name(e)); }
double seconds_since(std::chrono::steady_clock::time_point t0) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }

struct Ctx {
  fs::path data, results;
  int threads = 1;
  fs::path dir() const { return data / "l_swirl"; }
};

template <class F>
void parallel(int threads, int n, F&& f) {
  std::atomic<int> next{0};
  std::vector<std::jthread> pool;
  for (int t = 0; t < std::max(1, threads); ++t) {
    pool.emplace_back([&] {
      for (int i; (i = next++) < n;) f(i);
    });
  }
}

sim::Effect effect_of(const std::string& s) {
  sim::Effect e{};
  if (!sim::parse_effect(s, e)) throw std::invalid_argument("unknown effect " + s);
  return e;
}

rollout::Model load_or_throw(const fs::path& p) {
  auto m = rollout::load_model(p);
  if (!m) throw std::runtime_error(std::format("{}: {}", p.string(), m.error()));
  return std::move(*m);
}

double apsnr(std::span<const std::uint8_t> r, std::span<const std::uint8_t> t) {  // as study D's d-eval
  double sum = 0, n = 0;
  for (std::size_t i = 0; i < r.size(); i += 4) {
    if (std::max({r[i], r[i + 1], r[i + 2], r[i + 3]}) <= 4 && std::max({t[i], t[i + 1], t[i + 2], t[i + 3]}) <= 4) continue;
    n += 4;
    for (std::size_t c = 0; c < 4; ++c) {
      const double d = (static_cast<double>(r[i + c]) - static_cast<double>(t[i + c])) / 255.0;
      sum += d * d;
    }
  }
  return n > 0 ? metrics::psnr_from_mse(sum / n) : metrics::kPsnrCap;
}

// --- the runtime ----------------------------------------------------------------------------------------------------------

struct Runtime {
  nvfx_effect* e = nullptr;
  explicit Runtime(const rollout::Model& m) {
    std::ostringstream os;
    if (auto r = rollout::save_model(os, m); !r) throw std::runtime_error(r.error());
    const std::string b = os.str();
    if (nvfx_effect_load_memory(b.data(), b.size(), &e) != NVFX_OK) throw std::runtime_error("runtime load failed");
  }
  ~Runtime() { nvfx_effect_free(e); }
  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;
};

// `frames` frames of the runtime at controls: a start point (variation >= 0, which keeps its run's seed) or a seed;
// shards (as shipped) or one continuous rollout (tracking).
Clip runtime_clip(nvfx_effect* e, std::span<const float> controls, int variation, std::uint64_t seed, int frames, bool shards = true) {
  nvfx_instance* in = nullptr;
  if (nvfx_instance_create(e, kSize, &in) != NVFX_OK) throw std::runtime_error("instance");
  nvfx_instance_set_controls(in, controls.data(), static_cast<int>(controls.size()));
  if (!shards) nvfx_instance_set_drift(in, 0.f);
  if (variation >= 0) nvfx_instance_set_variation(in, variation);
  else nvfx_instance_set_seed(in, seed);
  Clip c;
  c.allocate(kSize, frames);
  c.fps = 30.f;
  for (int f = 0; f < frames; ++f) nvfx_render(in, f / 30.0, c.frame(f).data(), kSize * 4);
  nvfx_instance_free(in);
  return c;
}

// The simulation's own renderer on heat and soot fields at 128 x 128.
void render_fields(const sim::Params& p, const std::vector<float>& T, const std::vector<float>& D, std::span<std::uint8_t> out) {
  sim::Params q = p;
  q.sim_res = kSize;
  q.size = kSize;
  sim::Fluid f(q);
  sim::State s;
  s.n = kSize;
  s.u.assign(T.size(), 0.f);
  s.v.assign(T.size(), 0.f);
  s.pressure.assign(T.size(), 0.f);
  s.temp = T;
  s.soot = D;
  f.set_state(s);
  f.render(out);
}

// Bilinear upsampling of a coarse channel (as the runtime's start without fine fields).
std::vector<float> upsample(const std::vector<float>& coarse, int channels, int c) {
  std::vector<float> o(sz(kSize) * sz(kSize));
  const float k = fl(kSize) / fl(kRes);
  for (int y = 0; y < kSize; ++y) {
    for (int x = 0; x < kSize; ++x) {
      float xs = std::clamp((fl(x) + 0.5f) / k - 0.5f, 0.f, fl(kRes - 1)), ys = std::clamp((fl(y) + 0.5f) / k - 0.5f, 0.f, fl(kRes - 1));
      const int x0 = std::min(static_cast<int>(xs), kRes - 2), y0 = std::min(static_cast<int>(ys), kRes - 2);
      const float fx = xs - fl(x0), fy = ys - fl(y0);
      const auto at = [&](int xx, int yy) { return coarse[(sz(yy) * sz(kRes) + sz(xx)) * sz(channels) + sz(c)]; };
      o[sz(y) * sz(kSize) + sz(x)] = (1.f - fy) * ((1.f - fx) * at(x0, y0) + fx * at(x0 + 1, y0)) + fy * ((1.f - fx) * at(x0, y0 + 1) + fx * at(x0 + 1, y0 + 1));
    }
  }
  return o;
}

// --- detail layer variants (diagnosis only; the reference is rollout::detail_step) ------------------------------------------

// What a variant changes in the detail layer, beyond the constants of rollout::DetailSpec.
struct XOpts {
  bool rk2 = false;    // midpoint (second-order) back-trace in both MacCormack passes: rotation-aware
  int substeps = 1;    // the advection split into this many passes, each with the flow / substeps
  int streak = 0;      // the breakup noise of new material averaged at this many points along the frame's path
  float relax = 1.f;   // the lock moves the fine fields this fraction of the way to their locked values (1: all of it)
  bool sublock = false;  // with substeps: lock after every pass (new material arrives in smaller, closer steps)
  bool plain = true;   // no option set: rollout::detail_step itself
};

float bilin(const float* f, int n, float x, float y) {
  x = std::clamp(x, 0.f, fl(n - 1));
  y = std::clamp(y, 0.f, fl(n - 1));
  const int x0 = std::min(static_cast<int>(x), n - 2), y0 = std::min(static_cast<int>(y), n - 2);
  const float fx = x - fl(x0), fy = y - fl(y0);
  const float* a = f + sz(y0) * sz(n) + sz(x0);
  return (1.f - fy) * ((1.f - fx) * a[0] + fx * a[1]) + fy * ((1.f - fx) * a[n] + fx * a[n + 1]);
}
float bilin_ch(const float* f, int n, int channels, int c, float x, float y) {
  x = std::clamp(x, 0.f, fl(n - 1));
  y = std::clamp(y, 0.f, fl(n - 1));
  const int x0 = std::min(static_cast<int>(x), n - 2), y0 = std::min(static_cast<int>(y), n - 2);
  const float fx = x - fl(x0), fy = y - fl(y0);
  const auto at = [&](int xx, int yy) { return f[(sz(yy) * sz(n) + sz(xx)) * sz(channels) + sz(c)]; };
  return (1.f - fy) * ((1.f - fx) * at(x0, y0) + fx * at(x0 + 1, y0)) + fy * ((1.f - fx) * at(x0, y0 + 1) + fx * at(x0 + 1, y0 + 1));
}
float bilin_zero(const float* f, int n, float x, float y, float* lo = nullptr, float* hi = nullptr) {
  x = std::clamp(x, -1.f, fl(n));
  y = std::clamp(y, -1.f, fl(n));
  const float fx0 = std::floor(x), fy0 = std::floor(y);
  const int x0 = static_cast<int>(fx0), y0 = static_cast<int>(fy0);
  const float fx = x - fx0, fy = y - fy0;
  const auto at = [&](int xx, int yy) { return (xx < 0 || yy < 0 || xx >= n || yy >= n) ? 0.f : f[sz(yy) * sz(n) + sz(xx)]; };
  const float a = at(x0, y0), b = at(x0 + 1, y0), c = at(x0, y0 + 1), d = at(x0 + 1, y0 + 1);
  if (lo) *lo = std::min(std::min(a, b), std::min(c, d));
  if (hi) *hi = std::max(std::max(a, b), std::max(c, d));
  return (1.f - fy) * ((1.f - fx) * a + fx * b) + fy * ((1.f - fx) * c + fx * d);
}

// The sub-grid swirl lattice of rollout.cpp (copied: diagnosis only).
struct Swirl {
  int n = 0;
  float spacing = 1;
  std::vector<float> u, v;
};
Swirl swirl_field(const rollout::Model& m, std::uint64_t seed, float t) {
  Swirl s;
  const rollout::DetailSpec& d = m.detail;
  s.spacing = 0.5f * d.swirl_scale;
  s.n = static_cast<int>(std::ceil(130.f / s.spacing)) + 3;
  const int n = s.n;
  std::vector<float> psi(sz(n) * sz(n));
  const std::uint64_t sseed = seed * 0x9E3779B97F4A7C15ULL + 5;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) psi[sz(j) * sz(n) + sz(i)] = value_noise(fl(i - 1) * 0.5f, fl(j - 1) * 0.5f, t * d.swirl_rate, sseed);
  }
  s.u.assign(psi.size(), 0.f);
  s.v.assign(psi.size(), 0.f);
  for (int j = 1; j < n - 1; ++j) {
    for (int i = 1; i < n - 1; ++i) {
      const std::size_t k = sz(j) * sz(n) + sz(i);
      s.u[k] = psi[k + sz(n)] - psi[k - sz(n)];
      s.v[k] = psi[k - 1] - psi[k + 1];
    }
  }
  return s;
}

float smooth01(float t) {
  t = std::clamp(t, 0.f, 1.f);
  return t * t * (3.f - 2.f * t);
}

// rollout::detail_step with the options of XOpts (with all of them off it computes the same as the reference).
void detail_step_x(const rollout::Model& m, rollout::State& s, std::uint64_t seed, std::span<const float> controls, const XOpts& xo) {
  if (xo.plain) {
    rollout::detail_step(m, s, seed, controls);
    return;
  }
  const rollout::DetailSpec& dt = m.detail;
  if (dt.soften > 0.f) {
    rollout::soften_field(s.fine_t, s.size, dt.soften);
    rollout::soften_field(s.fine_d, s.size, dt.soften);
  }
  const int R = m.h.res, S = s.size, C = m.h.channels();
  const float k = fl(S) / fl(R), px128 = fl(S) / 128.f, t = s.time + 0.5f / m.fps;
  const float ka = k * dt.advect / fl(xo.substeps);
  float amp = dt.swirl * px128 / fl(xo.substeps);
  if (dt.swirl_control >= 0 && sz(dt.swirl_control) < controls.size()) amp *= 0.3f + controls[sz(dt.swirl_control)];
  if (dt.swirl_ramp > 0.f) amp *= std::min(1.f, s.since_start / dt.swirl_ramp);
  Swirl sw;
  if (amp > 0.f) sw = swirl_field(m, seed, t);
  // the fine velocity at a continuous fine position (pixel centres at integers), pixels per pass
  const auto vel = [&](float px, float py, float& u, float& v) {
    const float xc = (px + 0.5f) / k - 0.5f, yc = (py + 0.5f) / k - 0.5f;
    u = bilin_ch(s.flow.data(), R, 2, 0, xc, yc) * ka;
    v = bilin_ch(s.flow.data(), R, 2, 1, xc, yc) * ka;
    if (amp > 0.f) {
      const float lx = ((px + 0.5f) / px128 + 0.5f) / sw.spacing + 1.f, ly = ((py + 0.5f) / px128 + 0.5f) / sw.spacing + 1.f;
      u += amp * bilin(sw.u.data(), sw.n, lx, ly);
      v += amp * bilin(sw.v.data(), sw.n, lx, ly);
    }
  };
  // where a pixel's material came from (dir -1) or goes to (dir +1) in one pass
  const auto trace = [&](int x, int y, float dir, float& px, float& py) {
    float u, v;
    vel(fl(x), fl(y), u, v);
    if (xo.rk2) {
      float um, vm;
      vel(fl(x) + 0.5f * dir * u, fl(y) + 0.5f * dir * v, um, vm);
      u = um;
      v = vm;
    }
    px = fl(x) + dir * u;
    py = fl(y) + dir * v;
  };
  const std::size_t N = sz(S) * sz(S);
  std::vector<float> fa(N), fb(N), lo(N), hi(N), bx(N), by(N), fx(N), fy(N);
  for (int y = 0; y < S; ++y) {
    for (int x = 0; x < S; ++x) {
      const std::size_t i = sz(y) * sz(S) + sz(x);
      trace(x, y, -1.f, bx[i], by[i]);
      trace(x, y, +1.f, fx[i], fy[i]);
    }
  }
  const int kk = S / R;
  std::vector<float> B(sz(R) * sz(R)), rr(sz(R) * sz(R)), aa(sz(R) * sz(R));
  const int locks = xo.sublock ? xo.substeps : 1;
  for (int lk = 0; lk < locks; ++lk) {
  for (int pass = 0; pass < (xo.sublock ? 1 : xo.substeps); ++pass) {
    for (std::vector<float>* q : {&s.fine_t, &s.fine_d}) {
      std::vector<float>& Q = *q;
      for (std::size_t i = 0; i < N; ++i) fa[i] = bilin_zero(Q.data(), S, bx[i], by[i], &lo[i], &hi[i]);
      for (std::size_t i = 0; i < N; ++i) fb[i] = bilin_zero(fa.data(), S, fx[i], fy[i]);
      for (std::size_t i = 0; i < N; ++i) Q[i] = std::max(0.f, std::clamp(fa[i] + 0.5f * (Q[i] - fb[i]), lo[i], hi[i]));
    }
  }
  for (int ch = 0; ch < 2; ++ch) {
    std::vector<float>& Q = ch == 0 ? s.fine_t : s.fine_d;
    std::ranges::fill(B, 0.f);
    for (int y = 0; y < S; ++y) {
      for (int x = 0; x < S; ++x) B[sz(y / kk) * sz(R) + sz(x / kk)] += Q[sz(y) * sz(S) + sz(x)];
    }
    for (int i = 0; i < R * R; ++i) {
      const float b = B[sz(i)] / fl(kk * kk), target = s.coarse[sz(i) * sz(C) + 2 + sz(ch)];
      constexpr float eps = 1e-4f;
      const float r = (target + eps) / (b + eps);
      rr[sz(i)] = r <= 1.f ? r : std::min(r, dt.grow);
      aa[sz(i)] = std::max(0.f, target - b * rr[sz(i)]);
    }
    for (int y = 0; y < S; ++y) {
      for (int x = 0; x < S; ++x) {
        const float xc = (fl(x) + 0.5f) / k - 0.5f, yc = (fl(y) + 0.5f) / k - 0.5f;
        const std::size_t i = sz(y) * sz(S) + sz(x);
        float add = bilin(aa.data(), R, xc, yc);
        if (add > 0.f && dt.contrast > 0.f) {
          float g = 0.f;
          const int np = std::max(1, xo.streak);
          for (int j = 0; j < np; ++j) {  // along the frame's path back from the pixel (one point: the pixel itself)
            const float a = xo.streak > 0 ? (fl(j) + 0.5f) / fl(np) * fl(xo.substeps) : 0.f;
            const float qx = fl(x) + a * (bx[i] - fl(x)), qy = fl(y) + a * (by[i] - fl(y));
            const float X = (qx + 0.5f) / px128 + 0.5f, Y = (qy + 0.5f) / px128 + 0.5f;
            const float phi = rollout::noise_flicker(m.noise, seed, X, Y, t);
            g += smooth01((phi - dt.edge0) / (dt.edge1 - dt.edge0)) / fl(np);
          }
          add *= (1.f - dt.contrast) + dt.contrast * dt.kappa * g;
        }
        const float scale = bilin(rr.data(), R, xc, yc);
        Q[i] = Q[i] * (1.f + xo.relax * (scale - 1.f)) + xo.relax * add;
      }
    }
  }
  }
}

// --- constants and variants by name ----------------------------------------------------------------------------------------

float contrast_kappa(const rollout::Model& m) {  // as rollout::calibrate_detail
  double mean = 0;
  constexpr int n = 20000;
  for (int i = 0; i < n; ++i) {
    const float phi = rollout::noise_flicker(m.noise, 9, static_cast<float>(i % 137) * 0.7f, static_cast<float>(i / 137) * 0.9f, 0.3f);
    const float t = std::clamp((phi - m.detail.edge0) / (m.detail.edge1 - m.detail.edge0), 0.f, 1.f);
    mean += static_cast<double>(t * t * (3.f - 2.f * t));
  }
  return static_cast<float>(n / std::max(1e-6, mean));
}

// A variant: "name:k=v;k=v" with DetailSpec constants (contrast, edge0, edge1, swirl, swirl_scale, swirl_rate, grow,
// advect, soften) and the options of XOpts (rk2, substeps, streak).
struct Variant {
  std::string name;
  rollout::Model m;
  XOpts xo;
};
Variant make_variant(const rollout::Model& base, const std::string& spec) {
  Variant v;
  const std::size_t colon = spec.find(':');
  v.name = spec.substr(0, colon);
  v.m = base;
  bool edges = false;
  if (colon != std::string::npos) {
    std::size_t a = colon + 1;
    const std::string text = spec;
    while (a < text.size()) {
      std::size_t b = text.find(';', a);
      if (b == std::string::npos) b = text.size();
      const std::string kv = text.substr(a, b - a);
      const std::size_t eq = kv.find('=');
      if (eq == std::string::npos) throw std::invalid_argument("variant: " + kv);
      const std::string key = kv.substr(0, eq);
      const float val = std::stof(kv.substr(eq + 1));
      rollout::DetailSpec& d = v.m.detail;
      if (key == "contrast") d.contrast = val;
      else if (key == "edge0") d.edge0 = val, edges = true;
      else if (key == "edge1") d.edge1 = val, edges = true;
      else if (key == "swirl") d.swirl = val;
      else if (key == "swirl_scale") d.swirl_scale = val;
      else if (key == "swirl_rate") d.swirl_rate = val;
      else if (key == "grow") d.grow = val;
      else if (key == "advect") d.advect = val;
      else if (key == "soften") d.soften = val;
      else if (key == "passes") d.passes = static_cast<int>(val);
      else if (key == "rk2") v.xo.rk2 = val != 0.f, v.xo.plain = false;
      else if (key == "substeps") v.xo.substeps = std::max(1, static_cast<int>(val)), v.xo.plain = false;
      else if (key == "streak") v.xo.streak = std::max(0, static_cast<int>(val)), v.xo.plain = false;
      else if (key == "relax") v.xo.relax = val, v.xo.plain = false;
      else if (key == "sublock") v.xo.sublock = val != 0.f, v.xo.plain = false;
      else throw std::invalid_argument("variant: unknown key " + key);
      a = b + 1;
    }
  }
  if (edges) v.m.detail.kappa = contrast_kappa(v.m);
  return v;
}
std::vector<std::string> split_list(const std::string& s, char sep) {
  std::vector<std::string> v;
  std::size_t a = 0;
  while (a <= s.size()) {
    const std::size_t b = std::min(s.find(sep, a), s.size());
    if (b > a) v.push_back(s.substr(a, b - a));
    a = b + 1;
  }
  return v;
}

// --- diag: the tracking error, taken apart (training runs) ------------------------------------------------------------------

const std::vector<int> kHorizons = {1, 4, 8, 16, 30, 60, 120, 240};

sim::Params run_params(sim::Effect e, std::uint64_t salt, std::uint64_t index) {
  rollout::SimRecipe r;
  r.effect = e;
  r.salt = salt;
  return rollout::recipe_run(r, index);
}

void write_rows(const fs::path& path, const std::string& header, const std::vector<std::string>& rows) {
  fs::create_directories(path.parent_path());
  std::ofstream o(path);
  o << header << "\n";
  for (const auto& r : rows) o << r << "\n";
}

void step_diag(const Ctx& c, sim::Effect e, const tools::Args& a) {
  const rollout::Model M = load_or_throw(a.need("model"));
  const int n_runs = a.i("runs", 8), first = a.i("first", 0), KE = a.i("frames", 240), warm = 100;
  const std::uint64_t salt = a.u64("salt", 1);
  if (salt == 2) throw std::invalid_argument("diag: salt 2 holds study D's test runs");
  const std::string tag = a.str("tag", "diag");
  std::vector<Variant> variants = {make_variant(M, "current")};
  for (const auto& v : split_list(a.str("variants", ""), '/')) variants.push_back(make_variant(M, v));
  const bool neural = !a.flag("no-neural");
  // methods: neural (runtime), neural dynamics drawn by the simulation's renderer, the neural coarse state alone
  // (upsampled), the true coarse state alone (upsampled), the true coarse state with each detail variant (the oracle
  // stepper), the coarse simulation with and without the detail layer, the learned renderer on the true fields, the
  // first frame held
  std::vector<std::string> methods;
  if (neural) methods = {"neural", "neural_simrender", "neural_coarse_up"};
  for (const auto& v : variants) {
    if (neural && v.name != "current") methods.push_back("neural_simrender_" + v.name);
  }
  const bool only_neural = a.flag("only-neural");
  if (!only_neural) {
    methods.push_back("oracle_up");
    for (const auto& v : variants) methods.push_back("oracle_detail_" + v.name);
    methods.insert(methods.end(), {"coarse_sim_detail", "coarse_sim", "renderer_true", "frozen"});
  }
  const std::size_t NM = methods.size();
  std::vector<std::vector<std::vector<double>>> curves(NM, std::vector<std::vector<double>>(sz(n_runs)));
  std::vector<Clip> shown;  // run 0's clips, for the figure
  Clip shown_ref;
  const auto t0 = std::chrono::steady_clock::now();
  parallel(c.threads, n_runs, [&](int r) {
    const sim::Params p = run_params(e, salt, static_cast<std::uint64_t>(first + r));
    const std::vector<float> ctl{p.intensity, p.wind, p.turbulence};
    sim::Fluid truth(p);
    for (int i = 0; i < warm; ++i) truth.step_frame();
    const sim::State st = truth.state();
    Clip ref;
    ref.allocate(kSize, KE);
    std::vector<std::vector<float>> true_coarse(sz(KE));  // [frame] res^2 * kPhys after each frame
    std::vector<std::vector<float>> true_t(sz(KE)), true_d(sz(KE));
    for (int i = 0; i < KE; ++i) {
      truth.step_frame();
      truth.render(ref.frame(i));
      const sim::State s2 = truth.state();
      true_coarse[sz(i)].resize(sz(kRes) * kRes * rollout::kPhys);
      rollout::coarse_from_sim(s2, kRes, p.fps, true_coarse[sz(i)]);
      if (i < 30) {
        true_t[sz(i)] = s2.temp;
        true_d[sz(i)] = s2.soot;
      }
    }
    std::vector<Clip> out(NM);
    for (auto& cl : out) cl.allocate(kSize, KE);
    const auto idx = [&](const std::string& name) { return static_cast<std::size_t>(std::ranges::find(methods, name) - methods.begin()); };
    // the true start point with its fine fields
    rollout::Model mt = M;
    mt.h.start_fine = kSize;
    rollout::StartPoint sp;
    sp.controls = ctl;
    sp.seed = p.seed;
    sp.time = st.time;
    sp.coarse.resize(sz(kRes) * kRes * rollout::kPhys);
    rollout::coarse_from_sim(st, kRes, p.fps, sp.coarse);
    sp.fine_t = st.temp;
    sp.fine_d = st.soot;
    mt.starts = {sp};
    if (neural) {
      Runtime fx(mt);
      const Clip nc = runtime_clip(fx.e, ctl, 0, 0, KE + 1, false);
      out[idx("neural")] = slice_clip(nc, 1, KE);
      for (const auto& v : variants) {
        rollout::Model mv = v.m;
        mv.h.start_fine = kSize;
        mv.starts = {sp};
        rollout::State rs = rollout::start(mv, 0, kSize, ctl, sp.seed);
        const bool cur = v.name == "current";
        const std::size_t k = idx(cur ? "neural_simrender" : "neural_simrender_" + v.name);
        for (int i = 0; i < KE; ++i) {
          // rollout::step with the variant's detail layer
          std::vector<float> cond(sz(mv.h.cond())), noise(sz(kRes) * kRes * rollout::kNoise), next(rs.coarse.size());
          rollout::condition(mv, ctl, rs.time, cond);
          rollout::coarse_noise(mv, sp.seed, rs.time, noise);
          rollout::coarse_step(mv, rs.coarse, noise, cond, rs.pressure, next, rs.flow);
          rs.coarse.swap(next);
          detail_step_x(mv, rs, sp.seed, ctl, v.xo);
          rs.time += 1.f / mv.fps;
          rs.since_start += 1.f / mv.fps;
          render_fields(p, rs.fine_t, rs.fine_d, out[k].frame(i));
          if (cur) render_fields(p, upsample(rs.coarse, mv.h.channels(), 2), upsample(rs.coarse, mv.h.channels(), 3), out[idx("neural_coarse_up")].frame(i));
        }
      }
    }
    // the oracle stepper: the true coarse state of every frame, its velocity moving the fine fields
    for (int i = 0; i < KE && !only_neural; ++i) {
      render_fields(p, upsample(true_coarse[sz(i)], rollout::kPhys, 2), upsample(true_coarse[sz(i)], rollout::kPhys, 3), out[idx("oracle_up")].frame(i));
    }
    for (const auto& v : variants) {
      if (only_neural) break;
      rollout::State rs;
      rs.res = kRes;
      rs.size = kSize;
      rs.time = st.time;
      rs.since_start = 0.f;
      const int C = v.m.h.channels();
      rs.coarse.assign(sz(kRes) * kRes * sz(C), 0.f);
      rs.flow.assign(sz(kRes) * kRes * 2, 0.f);
      rs.pressure.assign(sz(kRes) * kRes, 0.f);
      rs.fine_t = st.temp;
      rs.fine_d = st.soot;
      for (int i = 0; i < KE; ++i) {
        for (int j = 0; j < kRes * kRes; ++j) {
          for (int q = 0; q < rollout::kPhys; ++q) rs.coarse[sz(j) * sz(C) + sz(q)] = true_coarse[sz(i)][sz(j) * rollout::kPhys + sz(q)];
          rs.flow[sz(j) * 2] = true_coarse[sz(i)][sz(j) * rollout::kPhys];
          rs.flow[sz(j) * 2 + 1] = true_coarse[sz(i)][sz(j) * rollout::kPhys + 1];
        }
        detail_step_x(v.m, rs, p.seed, ctl, v.xo);
        rs.time += 1.f / v.m.fps;
        rs.since_start += 1.f / v.m.fps;
        render_fields(p, rs.fine_t, rs.fine_d, out[idx("oracle_detail_" + v.name)].frame(i));
      }
    }
    if (!only_neural) {  // the coarse simulation with and without the detail layer (as d-eval's coarse_sim_detail)
      sim::Params q = p;
      q.sim_res = kRes;
      sim::Fluid f(q);
      sim::State cs;
      cs.n = kRes;
      cs.frame = st.frame;
      cs.time = st.time;
      const auto n2 = sz(kRes) * kRes;
      for (auto* v : {&cs.u, &cs.v, &cs.temp, &cs.soot, &cs.pressure}) v->assign(n2, 0.f);
      const int kb = st.n / kRes;
      const float ia = 1.f / fl(kb * kb);
      for (int y = 0; y < st.n; ++y) {
        for (int x = 0; x < st.n; ++x) {
          const std::size_t i = sz(y) * sz(st.n) + sz(x), o = sz(y / kb) * kRes + sz(x / kb);
          cs.u[o] += st.u[i] * ia / fl(kb);
          cs.v[o] += st.v[i] * ia / fl(kb);
          cs.temp[o] += st.temp[i] * ia;
          cs.soot[o] += st.soot[i] * ia;
        }
      }
      f.set_state(cs);
      rollout::State rs;
      rs.res = kRes;
      rs.size = kSize;
      rs.time = st.time;
      rs.since_start = 0.f;
      const int C = M.h.channels();
      rs.coarse.assign(sz(kRes) * kRes * sz(C), 0.f);
      rs.flow.assign(sz(kRes) * kRes * 2, 0.f);
      rs.pressure.assign(sz(kRes) * kRes, 0.f);
      rs.fine_t = st.temp;
      rs.fine_d = st.soot;
      for (int i = 0; i < KE; ++i) {
        f.step_frame();
        const sim::State x = f.state();
        for (std::size_t j = 0; j < n2; ++j) {
          rs.flow[j * 2] = x.u[j] / p.fps;
          rs.flow[j * 2 + 1] = x.v[j] / p.fps;
          rs.coarse[j * sz(C) + 2] = x.temp[j];
          rs.coarse[j * sz(C) + 3] = x.soot[j];
        }
        rollout::detail_step(M, rs, p.seed, ctl);
        rs.time += 1.f / p.fps;
        rs.since_start += 1.f / p.fps;
        render_fields(p, rs.fine_t, rs.fine_d, out[idx("coarse_sim_detail")].frame(i));
        f.render(out[idx("coarse_sim")].frame(i));
      }
    }
    if (!only_neural) {  // the learned renderer on the true fields (first 30 frames; then the last of them repeated)
      std::vector<float> rgba(sz(kSize) * kSize * 4);
      for (int i = 0; i < KE; ++i) {
        const int j = std::min(i, 29);
        if (i > 29) {
          std::ranges::copy(out[idx("renderer_true")].frame(29), out[idx("renderer_true")].frame(i).begin());
          continue;
        }
        rollout::State rs;
        rs.res = kRes;
        rs.size = kSize;
        const int C = M.h.channels();
        rs.coarse.assign(sz(kRes) * kRes * sz(C), 0.f);
        for (int q = 0; q < kRes * kRes; ++q) {
          for (int k = 0; k < rollout::kPhys; ++k) rs.coarse[sz(q) * sz(C) + sz(k)] = true_coarse[sz(j)][sz(q) * rollout::kPhys + sz(k)];
        }
        rs.fine_t = true_t[sz(j)];
        rs.fine_d = true_d[sz(j)];
        rollout::render(M, rs, rgba);
        auto o = out[idx("renderer_true")].frame(i);
        for (std::size_t q = 0; q < rgba.size(); ++q) o[q] = static_cast<std::uint8_t>(rgba[q] * 255.f + 0.5f);
      }
    }
    if (!only_neural) {
      std::vector<std::uint8_t> fz(sz(kSize) * kSize * 4);
      sim::Fluid f0(p);
      f0.set_state(st);
      f0.render(fz);
      for (int i = 0; i < KE; ++i) std::ranges::copy(fz, out[idx("frozen")].frame(i).begin());
    }
    for (std::size_t mth = 0; mth < NM; ++mth) {
      for (int i = 0; i < KE; ++i) curves[mth][sz(r)].push_back(apsnr(ref.frame(i), out[mth].frame(i)));
    }
    if (r == 0) {
      shown_ref = ref;
      shown = {};
      for (auto& cl : out) shown.push_back(cl);
    }
    std::println("diag: {} run {} done ({:.0f} s)", ename(e), first + r, seconds_since(t0));
    std::fflush(stdout);
  });
  // table: mean over runs per horizon; rows to CSV
  std::vector<std::string> rows;
  std::string head = "method";
  for (const int hz : kHorizons) {
    if (hz <= KE) head += std::format(",f{}", hz);
  }
  std::println("\n{}: active PSNR (dB), mean over {} training runs (salt {}, from {})", ename(e), n_runs, salt, first);
  std::println("{:<34}{}", "method", [&] {
    std::string s;
    for (const int hz : kHorizons) {
      if (hz <= KE) s += std::format("{:>8}", hz);
    }
    return s;
  }());
  for (std::size_t mth = 0; mth < NM; ++mth) {
    std::string line = std::format("{:<34}", methods[mth]);
    for (const int hz : kHorizons) {
      if (hz > KE) continue;
      double sum = 0;
      for (int r = 0; r < n_runs; ++r) sum += curves[mth][sz(r)][sz(hz - 1)];
      line += std::format("{:>8.2f}", sum / n_runs);
    }
    std::println("{}", line);
    for (int r = 0; r < n_runs; ++r) {
      std::string row = std::format("{},{},{},{}", ename(e), tag, first + r, methods[mth]);
      for (const int hz : kHorizons) {
        if (hz <= KE) row += std::format(",{:.3f}", curves[mth][sz(r)][sz(hz - 1)]);
      }
      rows.push_back(row);
    }
  }
  // paired differences against the oracle with the current detail layer and against neural, at 1 s
  if (KE >= 30) {
    const std::size_t h = 29;
    const auto col = [&](std::size_t mth) {
      std::vector<double> v;
      for (int r = 0; r < n_runs; ++r) v.push_back(curves[mth][sz(r)][h]);
      return v;
    };
    const std::size_t base = only_neural ? 0 : static_cast<std::size_t>(std::ranges::find(methods, "oracle_detail_current") - methods.begin());
    std::println("\nat 1 s, each method minus {} (paired over runs, 95% bootstrap):", methods[base]);
    for (std::size_t mth = 0; mth < NM; ++mth) {
      if (mth == base) continue;
      const auto iv = metrics::paired_bootstrap(col(mth), col(base), 10000, 7);
      std::println("  {:<34} {:+.2f} [{:+.2f}, {:+.2f}]", methods[mth], iv.mean, iv.lo, iv.hi);
    }
  }
  std::string h2 = "effect,tag,run,method";
  for (const int hz : kHorizons) {
    if (hz <= KE) h2 += std::format(",f{}", hz);
  }
  const fs::path csv = c.dir() / "diag" / std::format("{}_{}.csv", ename(e), tag);
  write_rows(csv, h2, rows);
  std::println("\nrows: {}", csv.string());
  if (a.has("figure")) {  // run 0: truth, then each method, at frames 1, 8, 30, 60, 120, 240
    std::vector<int> cols;
    for (const int f : {1, 8, 30, 60, 120, 240}) {
      if (f <= KE) cols.push_back(f - 1);
    }
    const auto pick = [&](const Clip& src) {
      Clip o;
      o.allocate(kSize, static_cast<int>(cols.size()));
      for (std::size_t q = 0; q < cols.size(); ++q) std::ranges::copy(src.frame(cols[q]), o.frame(static_cast<int>(q)).begin());
      return o;
    };
    std::vector<std::string> want = split_list(a.str("figure-methods", "neural,neural_simrender,oracle_detail_current,oracle_up,coarse_sim_detail"), ',');
    std::vector<Clip> picked = {pick(shown_ref)};
    for (const auto& w : want) {
      const auto it = std::ranges::find(methods, w);
      if (it != methods.end()) picked.push_back(pick(shown[static_cast<std::size_t>(it - methods.begin())]));
    }
    std::vector<const Clip*> rws;
    for (const auto& p : picked) rws.push_back(&p);
    if (auto w = write_png(a.str("figure"), comparison_sheet(rws, static_cast<int>(cols.size()), Background::black, a.i("scale", 1))); !w) {
      std::println("figure: {}", w.error());
    }
  }
}


// --- coarse: the stepper at the coarse level (training runs) ------------------------------------------------------------------

// Moments of a heat field on an n x n grid (cells from the bottom): total, mean radius from the middle (in fractions of the
// frame), and the share of heat within each of 8 radial bands.
struct Moments {
  double total = 0, radius = 0, cy = 0;
};
Moments moments(std::span<const float> heat, int n, int stride, int c) {
  Moments m;
  double sr = 0, sy = 0;
  for (int y = 0; y < n; ++y) {
    for (int x = 0; x < n; ++x) {
      const double h = std::max(0.f, heat[(sz(y) * sz(n) + sz(x)) * sz(stride) + sz(c)]);
      const double dx = (x + 0.5) / n - 0.5, dy = (y + 0.5) / n - 0.5;
      m.total += h;
      sr += h * std::sqrt(dx * dx + dy * dy);
      sy += h * (y + 0.5) / n;
    }
  }
  m.radius = m.total > 0 ? sr / m.total : 0;
  m.cy = m.total > 0 ? sy / m.total : 0;
  m.total /= static_cast<double>(n) * n;
  return m;
}
// Relative error of a 32-cell heat (or soot) field against the true one: |a - b|_2 / |b|_2.
double rel_err(std::span<const float> a, int sa, std::span<const float> b, int sb, int c) {
  double e = 0, t = 0;
  for (int i = 0; i < kRes * kRes; ++i) {
    const double x = a[sz(i) * sz(sa) + sz(c)], y = b[sz(i) * sz(sb) + sz(c)];
    e += (x - y) * (x - y);
    t += y * y;
  }
  return t > 0 ? std::sqrt(e / t) : 0;
}
std::vector<float> block_average(const sim::State& st, int res, float fps) {
  std::vector<float> o(sz(res) * sz(res) * rollout::kPhys);
  rollout::coarse_from_sim(st, res, fps, o);
  return o;
}

// The neural stepper's coarse state against the true one and against coarse simulations at 32, 48 and 64 cells (their
// heat block-averaged to 32), from true states: relative heat error by horizon, and the moments (total heat, mean radius)
// that show a bias.
void step_coarse(const Ctx& c, sim::Effect e, const tools::Args& a) {
  const rollout::Model M = load_or_throw(a.need("model"));
  const int n_runs = a.i("runs", 8), first = a.i("first", 0), KE = 240, warm = 100;
  const std::uint64_t salt = a.u64("salt", 1);
  const std::vector<std::string> methods = {"neural", "sim32", "sim48", "sim64", "persist"};
  const std::vector<int> H = {1, 8, 30, 60, 240};
  // [method][run][h]: relative heat error; and moments at h for truth and each method
  std::vector<std::vector<std::vector<double>>> err(methods.size(), std::vector<std::vector<double>>(sz(n_runs)));
  std::vector<std::vector<std::vector<Moments>>> mom(methods.size() + 1, std::vector<std::vector<Moments>>(sz(n_runs)));
  std::vector<std::vector<double>> sim_track(3, std::vector<double>(sz(n_runs)));  // coarse sims at 32, 48, 64: active PSNR at 1 s (own renderer)
  const auto t0 = std::chrono::steady_clock::now();
  parallel(c.threads, n_runs, [&](int r) {
    const sim::Params p = run_params(e, salt, static_cast<std::uint64_t>(first + r));
    const std::vector<float> ctl{p.intensity, p.wind, p.turbulence};
    sim::Fluid truth(p);
    for (int i = 0; i < warm; ++i) truth.step_frame();
    const sim::State st = truth.state();
    std::vector<std::vector<float>> tc(sz(KE));
    std::vector<std::vector<std::uint8_t>> ref(sz(KE), std::vector<std::uint8_t>(sz(kSize) * kSize * 4));
    for (int i = 0; i < KE; ++i) {
      truth.step_frame();
      tc[sz(i)] = block_average(truth.state(), kRes, p.fps);
      truth.render(ref[sz(i)]);
    }
    const std::vector<float> c0 = block_average(st, kRes, p.fps);
    const auto record = [&](std::size_t m, int i, std::span<const float> f, int stride) {
      for (const int h : H) {
        if (i + 1 == h) {
          err[m][sz(r)].push_back(rel_err(f, stride, tc[sz(i)], rollout::kPhys, 2));
          mom[m][sz(r)].push_back(moments(f, kRes, stride, 2));
        }
      }
    };
    for (int i = 0; i < KE; ++i) {
      for (const int h : H) {
        if (i + 1 == h) mom[methods.size()][sz(r)].push_back(moments(tc[sz(i)], kRes, rollout::kPhys, 2));
      }
      record(4, i, c0, rollout::kPhys);
    }
    {  // the neural stepper (reference), from the true coarse state
      const int C = M.h.channels();
      std::vector<float> S(sz(kRes) * kRes * sz(C), 0.f), next(S.size()), pressure(sz(kRes) * kRes, 0.f), flow(sz(kRes) * kRes * 2), cond(sz(M.h.cond())),
          noise(sz(kRes) * kRes * rollout::kNoise);
      for (int q = 0; q < kRes * kRes; ++q) {
        for (int k = 0; k < rollout::kPhys; ++k) S[sz(q) * sz(C) + sz(k)] = c0[sz(q) * rollout::kPhys + sz(k)];
      }
      float t = st.time;
      for (int i = 0; i < KE; ++i) {
        rollout::condition(M, ctl, t, cond);
        rollout::coarse_noise(M, p.seed, t, noise);
        rollout::coarse_step(M, S, noise, cond, pressure, next, flow);
        S.swap(next);
        t += 1.f / M.fps;
        record(0, i, S, C);
      }
    }
    for (int g = 0; g < 3; ++g) {  // coarse simulations from the block-averaged true state
      const int n = g == 0 ? 32 : g == 1 ? 48 : 64;
      sim::Params q = p;
      q.sim_res = n;
      sim::Fluid f(q);
      sim::State cs;
      cs.n = n;
      cs.frame = st.frame;
      cs.time = st.time;
      const auto n2 = sz(n) * sz(n);
      for (auto* v : {&cs.u, &cs.v, &cs.temp, &cs.soot, &cs.pressure}) v->assign(n2, 0.f);
      const int kb = st.n / n;  // 4, 2 (48: not a divisor of 128; the nearest cells are averaged), 2
      for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
          // average the fine cells whose centres fall in this coarse cell
          const int x0 = x * st.n / n, x1 = (x + 1) * st.n / n, y0 = y * st.n / n, y1 = (y + 1) * st.n / n;
          double su = 0, sv = 0, stt = 0, ss = 0;
          int cnt = 0;
          for (int yy = y0; yy < y1; ++yy) {
            for (int xx = x0; xx < x1; ++xx) {
              const std::size_t i = sz(yy) * sz(st.n) + sz(xx);
              su += st.u[i];
              sv += st.v[i];
              stt += st.temp[i];
              ss += st.soot[i];
              ++cnt;
            }
          }
          const std::size_t o = sz(y) * sz(n) + sz(x);
          const double scale = static_cast<double>(n) / st.n;  // velocity in this grid's cells per second
          cs.u[o] = static_cast<float>(su / cnt * scale);
          cs.v[o] = static_cast<float>(sv / cnt * scale);
          cs.temp[o] = static_cast<float>(stt / cnt);
          cs.soot[o] = static_cast<float>(ss / cnt);
        }
      }
      (void)kb;
      f.set_state(cs);
      std::vector<std::uint8_t> fr(sz(kSize) * kSize * 4);
      for (int i = 0; i < KE; ++i) {
        f.step_frame();
        const sim::State x = f.state();
        // heat on the 32-cell grid: area-weighted average of this grid's cells
        std::vector<float> h32(sz(kRes) * kRes * rollout::kPhys, 0.f);
        for (int y = 0; y < kRes; ++y) {
          for (int xx = 0; xx < kRes; ++xx) {
            double sum = 0, wsum = 0;
            for (int yy = 0; yy < n; ++yy) {
              const double oy = std::max(0.0, std::min((y + 1.0) / kRes, (yy + 1.0) / n) - std::max(y / static_cast<double>(kRes), yy / static_cast<double>(n)));
              if (oy <= 0) continue;
              for (int x2 = 0; x2 < n; ++x2) {
                const double ox = std::max(0.0, std::min((xx + 1.0) / kRes, (x2 + 1.0) / n) - std::max(xx / static_cast<double>(kRes), x2 / static_cast<double>(n)));
                if (ox <= 0) continue;
                sum += ox * oy * x.temp[sz(yy) * sz(n) + sz(x2)];
                wsum += ox * oy;
              }
            }
            h32[(sz(y) * kRes + sz(xx)) * rollout::kPhys + 2] = static_cast<float>(sum / wsum);
          }
        }
        record(sz(1 + g), i, h32, rollout::kPhys);
        if (i == 29) {
          f.render(fr);
          sim_track[sz(g)][sz(r)] = apsnr(ref[29], fr);
        }
      }
    }
    std::println("coarse: {} run {} done ({:.0f} s)", ename(e), first + r, seconds_since(t0));
    std::fflush(stdout);
  });
  std::println("\n{}: relative error of the 32-cell heat against the truth's, mean over {} runs (salt {}, from {})", ename(e), n_runs, salt, first);
  std::println("{:<10}{:>8}{:>8}{:>8}{:>8}{:>8}", "method", "f1", "f8", "1s", "2s", "8s");
  for (std::size_t m = 0; m < methods.size(); ++m) {
    std::string line = std::format("{:<10}", methods[m]);
    for (std::size_t h = 0; h < H.size(); ++h) {
      double s2 = 0;
      for (int r = 0; r < n_runs; ++r) s2 += err[m][sz(r)][h] / n_runs;
      line += std::format("{:>8.3f}", s2);
    }
    std::println("{}", line);
  }
  std::println("\nmoments at each horizon (mean over runs): total heat per cell / mean radius (frame fractions) / centre height");
  for (std::size_t m = 0; m <= methods.size(); ++m) {
    if (m == 4) continue;
    std::string line = std::format("{:<10}", m == methods.size() ? "truth" : methods[m]);
    for (std::size_t h = 0; h < H.size(); ++h) {
      double tt = 0, rr = 0, yy = 0;
      for (int r = 0; r < n_runs; ++r) {
        tt += mom[m][sz(r)][h].total / n_runs;
        rr += mom[m][sz(r)][h].radius / n_runs;
        yy += mom[m][sz(r)][h].cy / n_runs;
      }
      line += std::format("  {:.3f}/{:.3f}/{:.3f}", tt, rr, yy);
    }
    std::println("{}", line);
  }
  std::println("\ncoarse simulations, own renderer, active PSNR at 1 s: 32 cells {:.2f}, 48 cells {:.2f}, 64 cells {:.2f}",
               std::accumulate(sim_track[0].begin(), sim_track[0].end(), 0.0) / n_runs, std::accumulate(sim_track[1].begin(), sim_track[1].end(), 0.0) / n_runs,
               std::accumulate(sim_track[2].begin(), sim_track[2].end(), 0.0) / n_runs);
}

// --- advect: how much one coarse advection step loses (training runs) ------------------------------------------------------

// Bilinear sample of channel c of a [cell][stride] field, clamped to the grid (the stepper's semi-Lagrangian advection).
float samp(std::span<const float> f, int n, int stride, int c, float x, float y) { return bilin_ch(f.data(), n, stride, c, x, y); }

// One step of the true 32-cell heat by the true coarse velocity (the mean of the frame's two states), with the stepper's
// scheme (semi-Lagrangian, bilinear, a straight back-trace) and less diffusive ones, against the true next heat.
void step_advect(const Ctx& c, sim::Effect e, const tools::Args& a) {
  const int n_runs = a.i("runs", 8), frames = a.i("frames", 60);
  const std::vector<std::string> names = {"persist", "sl", "sl_rk2", "maccormack", "maccormack_rk2", "sl_bicubic_rk2"};
  std::vector<double> err(names.size(), 0.0), tot(names.size(), 0.0);
  std::mutex mu;
  parallel(c.threads, n_runs, [&](int r) {
    const sim::Params p = run_params(e, 1, static_cast<std::uint64_t>(r));
    sim::Fluid f(p);
    for (int i = 0; i < 100; ++i) f.step_frame();
    std::vector<float> prev = block_average(f.state(), kRes, p.fps);
    std::vector<double> le(names.size(), 0.0), lt(names.size(), 0.0);
    for (int i = 0; i < frames; ++i) {
      f.step_frame();
      const std::vector<float> next = block_average(f.state(), kRes, p.fps);
      const auto vel = [&](float x, float y, float& u, float& v) {
        u = 0.5f * (samp(prev, kRes, rollout::kPhys, 0, x, y) + samp(next, kRes, rollout::kPhys, 0, x, y));
        v = 0.5f * (samp(prev, kRes, rollout::kPhys, 1, x, y) + samp(next, kRes, rollout::kPhys, 1, x, y));
      };
      const auto back = [&](int x, int y, bool rk2, float dir, float& px, float& py) {
        float u, v;
        vel(fl(x), fl(y), u, v);
        if (rk2) vel(fl(x) + 0.5f * dir * u, fl(y) + 0.5f * dir * v, u, v);
        px = fl(x) + dir * u;
        py = fl(y) + dir * v;
      };
      const auto cubic = [&](std::span<const float> fld, float x, float y) {  // Catmull-Rom, clamped to the neighbours' range
        x = std::clamp(x, 0.f, fl(kRes - 1));
        y = std::clamp(y, 0.f, fl(kRes - 1));
        const int x0 = static_cast<int>(std::floor(x)), y0 = static_cast<int>(std::floor(y));
        const float tx = x - fl(x0), ty = y - fl(y0);
        const auto w = [](float t, int k) {
          switch (k) {
            case 0: return 0.5f * (-t + 2 * t * t - t * t * t);
            case 1: return 0.5f * (2 - 5 * t * t + 3 * t * t * t);
            case 2: return 0.5f * (t + 4 * t * t - 3 * t * t * t);
            default: return 0.5f * (-t * t + t * t * t);
          }
        };
        float sum = 0, lo = 1e30f, hi = -1e30f;
        for (int j = 0; j < 4; ++j) {
          const int yy = std::clamp(y0 - 1 + j, 0, kRes - 1);
          for (int i2 = 0; i2 < 4; ++i2) {
            const int xx = std::clamp(x0 - 1 + i2, 0, kRes - 1);
            const float val = fld[(sz(yy) * kRes + sz(xx)) * rollout::kPhys + 2];
            sum += w(tx, i2) * w(ty, j) * val;
            if ((i2 == 1 || i2 == 2) && (j == 1 || j == 2)) {
              lo = std::min(lo, val);
              hi = std::max(hi, val);
            }
          }
        }
        return std::clamp(sum, lo, hi);
      };
      for (std::size_t m = 0; m < names.size(); ++m) {
        std::vector<float> out(sz(kRes) * kRes);
        if (m == 0) {
          for (int q = 0; q < kRes * kRes; ++q) out[sz(q)] = prev[sz(q) * rollout::kPhys + 2];
        } else if (m == 1 || m == 2 || m == 5) {
          for (int y = 0; y < kRes; ++y) {
            for (int x = 0; x < kRes; ++x) {
              float px, py;
              back(x, y, m != 1, -1.f, px, py);
              out[sz(y) * kRes + sz(x)] = m == 5 ? cubic(prev, px, py) : samp(prev, kRes, rollout::kPhys, 2, px, py);
            }
          }
        } else {  // MacCormack, clamped to the forward sample's stencil
          const bool rk2 = m == 4;
          std::vector<float> fa(sz(kRes) * kRes), lo(fa.size()), hi(fa.size());
          for (int y = 0; y < kRes; ++y) {
            for (int x = 0; x < kRes; ++x) {
              float px, py;
              back(x, y, rk2, -1.f, px, py);
              px = std::clamp(px, 0.f, fl(kRes - 1));
              py = std::clamp(py, 0.f, fl(kRes - 1));
              const int x0 = std::min(static_cast<int>(px), kRes - 2), y0 = std::min(static_cast<int>(py), kRes - 2);
              float l = 1e30f, h = -1e30f;
              for (const int dy : {0, 1}) {
                for (const int dx : {0, 1}) {
                  const float val = prev[(sz(y0 + dy) * kRes + sz(x0 + dx)) * rollout::kPhys + 2];
                  l = std::min(l, val);
                  h = std::max(h, val);
                }
              }
              const std::size_t i2 = sz(y) * kRes + sz(x);
              fa[i2] = samp(prev, kRes, rollout::kPhys, 2, px, py);
              lo[i2] = l;
              hi[i2] = h;
            }
          }
          for (int y = 0; y < kRes; ++y) {
            for (int x = 0; x < kRes; ++x) {
              float px, py;
              back(x, y, rk2, +1.f, px, py);
              const std::size_t i2 = sz(y) * kRes + sz(x);
              const float fb = bilin(fa.data(), kRes, px, py);
              out[i2] = std::clamp(fa[i2] + 0.5f * (prev[i2 * rollout::kPhys + 2] - fb), lo[i2], hi[i2]);
            }
          }
        }
        for (int q = 0; q < kRes * kRes; ++q) {
          const double d = out[sz(q)] - next[sz(q) * rollout::kPhys + 2], t = next[sz(q) * rollout::kPhys + 2];
          le[m] += d * d;
          lt[m] += t * t;
        }
      }
      prev = next;
    }
    std::lock_guard lk(mu);
    for (std::size_t m = 0; m < names.size(); ++m) {
      err[m] += le[m];
      tot[m] += lt[m];
    }
  });
  std::println("{}: one coarse advection step of the true heat by the true velocity, relative error against the next true heat ({} runs x {} frames)", ename(e),
               n_runs, frames);
  for (std::size_t m = 0; m < names.size(); ++m) std::println("  {:<16} {:.4f}", names[m], std::sqrt(err[m] / tot[m]));
}

// --- graft: one model's stepper with another's renderer, detail layer and start points -------------------------------------

// A model with other detail constants (DetailSpec keys of make_variant), written as a new file.
void step_set(const tools::Args& a) {
  const rollout::Model m = load_or_throw(a.need("model"));
  const Variant v = make_variant(m, "set:" + a.need("detail"));
  if (!v.xo.plain) throw std::invalid_argument("set: only the file's own constants");
  if (auto w = rollout::save_model(a.need("out"), v.m); !w) throw std::runtime_error(w.error());
  std::println("set: {} with {} -> {}", a.need("model"), a.need("detail"), a.need("out"));
}

void step_graft(const tools::Args& a) {
  const rollout::Model st = load_or_throw(a.need("stepper"));
  rollout::Model m = load_or_throw(a.need("rest"));
  if (st.step_w.size() != m.step_w.size()) throw std::invalid_argument("graft: the steppers differ in shape");
  m.step_w = st.step_w;
  m.scale = st.scale;
  m.lo = st.lo;
  m.hi = st.hi;
  m.qscale = st.qscale;
  if (auto w = rollout::save_model(a.need("out"), m); !w) throw std::runtime_error(w.error());
  std::println("graft: {} with the stepper of {} -> {}", a.need("rest"), a.need("stepper"), a.need("out"));
}

// --- train: continue training a stepper (or the renderer) with other options ----------------------------------------------

// The recipe's 160 training runs (salt 1), recorded once and cached under DATA/l_swirl/runs; with --extra-low K, K more
// salt-1 runs (indices from 160, beyond the recipe's) whose intensity is at most --low-max (0.4).
std::vector<rollout::Run> training_runs(const Ctx& c, sim::Effect e, int extra_low, float low_max, int threads) {
  rollout::SimRecipe r = rollout::recipe_for(e);
  std::vector<std::uint64_t> idx;
  for (int i = 0; i < r.runs; ++i) idx.push_back(static_cast<std::uint64_t>(i));
  for (std::uint64_t i = static_cast<std::uint64_t>(r.runs); extra_low > 0 && i < 100000; ++i) {
    if (run_params(e, 1, i).intensity <= low_max) {
      idx.push_back(i);
      --extra_low;
    }
  }
  std::vector<rollout::Run> runs(idx.size());
  const std::size_t per = sz(kRes) * kRes * rollout::kPhys;
  parallel(threads, static_cast<int>(idx.size()), [&](int j) {
    const fs::path p = c.dir() / "runs" / ename(e) / std::format("run_{}.bin", idx[sz(j)]);
    rollout::Run& run = runs[sz(j)];
    run.p = run_params(e, 1, idx[sz(j)]);
    run.frames = r.frames;
    run.coarse.resize(sz(r.frames) * per);
    if (std::ifstream in(p, std::ios::binary); in) {
      in.read(reinterpret_cast<char*>(run.coarse.data()), static_cast<std::streamsize>(run.coarse.size() * sizeof(float)));
      if (in) return;
    }
    run = rollout::record_run(run.p, r.frames, kRes);
    fs::create_directories(p.parent_path());
    const fs::path tmp = p.string() + ".tmp";
    {
      std::ofstream o(tmp, std::ios::binary);
      o.write(reinterpret_cast<const char*>(run.coarse.data()), static_cast<std::streamsize>(run.coarse.size() * sizeof(float)));
    }
    fs::rename(tmp, p);
  });
  return runs;
}

// Study L (the vortex): every training run starts from a state after its first frame, so no window ever starts from an
// empty grid, and the stepper never learns to grow its column from nothing. With empty_start each run gets the empty grid
// as its state 0 (t0 one frame earlier: the step from it is the simulation's first frame); with growth > 0, that many
// more runs hold only the first `growth_frames` states of runs (the lowest intensities first), so windows in the growth
// from nothing are common.
std::vector<rollout::Run> with_growth(std::vector<rollout::Run> runs, bool empty_start, int growth, int growth_frames) {
  const std::size_t per = sz(kRes) * kRes * rollout::kPhys;
  if (empty_start || growth > 0) {
    for (auto& r : runs) {
      r.coarse.insert(r.coarse.begin(), per, 0.f);
      ++r.frames;
      r.t0 -= 1.f / r.p.fps;
    }
  }
  std::vector<std::size_t> order(runs.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::ranges::stable_sort(order, [&](std::size_t x, std::size_t y) { return runs[x].p.intensity < runs[y].p.intensity; });
  const std::size_t n = runs.size();
  for (int g = 0; g < growth; ++g) {
    rollout::Run r = runs[order[sz(g) % n]];
    r.frames = std::min(r.frames, growth_frames);
    r.coarse.resize(sz(r.frames) * per);
    runs.push_back(std::move(r));
  }
  return runs;
}

void step_train(const Ctx& c, sim::Effect e, const tools::Args& a) {
  rollout::Model m = load_or_throw(a.need("model"));
  if (a.has("advection")) m.h.advection = a.i("advection", 0);  // study L: MacCormack advection of the coarse state
  const fs::path out = a.need("out");
  const auto t0 = std::chrono::steady_clock::now();
  const auto runs = with_growth(training_runs(c, e, a.i("extra-low", 0), a.f("low-max", 0.4f), c.threads), a.flag("empty-start"), a.i("growth", 0),
                                a.i("growth-frames", 40));
  std::println("train: {} {} runs ready ({:.0f} s)", ename(e), runs.size(), seconds_since(t0));
  rollout::StepperOptions so = rollout::recipe_for(e).stepper;
  so.iterations = a.i("iterations", 0);
  so.finetune = a.i("finetune", 0);
  so.activity_stage = a.i("activity-stage", 0);
  so.activity = a.f("activity", 100.f);
  so.max_unroll = a.i("unroll", 16);
  so.burn_max = a.i("burn", 48);
  so.sigma = a.f("sigma", 0.03f);
  so.profile = a.f("profile", 1.f);
  so.lr = a.f("lr", 2e-3f);
  so.lr_finetune = a.f("lr-ft", 7e-4f);
  so.batch = a.i("batch", 16);
  so.seed = a.u64("seed", 11);
  so.keep_normalisation = true;
  so.threads = c.threads;
  so.log_every = 50;
  so.progress = [&](int it, int unroll, double loss) {
    std::println("  {} it {:5d} unroll {:2d} loss {:.5f} ({:.0f} s)", ename(e), it, unroll, loss, seconds_since(t0));
    std::fflush(stdout);
  };
  const int every = a.i("checkpoint-every", 0);
  if (every > 0) {
    so.checkpoint_every = every;
    so.checkpoint = [&](int done) {
      if (auto w = rollout::save_model(out.string() + std::format(".it{}", done), m); !w) throw std::runtime_error(w.error());
    };
  }
  const rollout::StepperResult r = rollout::train_stepper(m, runs, so);
  rollout::quantise_like_storage(m);
  if (auto w = rollout::save_model(out, m); !w) throw std::runtime_error(w.error());
  std::println("train: {} -> {}: loss {:.5f} in {:.0f} s", ename(e), out.string(), r.final_loss, r.seconds);
}

// --- statistics, cached -------------------------------------------------------------------------------------------------------

void put_vec(std::ostream& o, const std::vector<double>& v) {
  const auto n = static_cast<std::uint32_t>(v.size());
  o.write(reinterpret_cast<const char*>(&n), 4);
  o.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(double)));
}
bool get_vec(std::istream& i, std::vector<double>& v) {
  std::uint32_t n = 0;
  i.read(reinterpret_cast<char*>(&n), 4);
  if (!i || n > 100000) return false;
  v.resize(n);
  i.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(double)));
  return static_cast<bool>(i);
}
void save_stats(const fs::path& p, const metrics::ClipStats& s) {
  fs::create_directories(p.parent_path());
  const fs::path tmp = p.string() + ".tmp";
  {
    std::ofstream o(tmp, std::ios::binary);
    put_vec(o, s.coverage);
    put_vec(o, s.emission);
    put_vec(o, s.spectrum);
    std::vector<double> mf(s.mean_frame.begin(), s.mean_frame.end());
    put_vec(o, mf);
    o.write(reinterpret_cast<const char*>(&s.motion), sizeof(double));
  }
  fs::rename(tmp, p);
}
bool load_stats(const fs::path& p, metrics::ClipStats& s) {
  std::ifstream i(p, std::ios::binary);
  if (!i) return false;
  std::vector<double> mf;
  if (!get_vec(i, s.coverage) || !get_vec(i, s.emission) || !get_vec(i, s.spectrum) || !get_vec(i, mf)) return false;
  s.mean_frame.assign(mf.begin(), mf.end());
  i.read(reinterpret_cast<char*>(&s.motion), sizeof(double));
  return static_cast<bool>(i);
}

using Setting = std::array<float, 3>;
int frames_of(sim::Effect) { return 300; }
int warm_of(sim::Effect) { return 150; }
sim::Params setting_params(sim::Effect e, const Setting& s, std::uint64_t seed) {
  sim::Params q;
  q.effect = e;
  q.intensity = s[0];
  q.wind = s[1];
  q.turbulence = s[2];
  q.seed = seed;
  q.size = kSize;
  return q;
}
// A real run's statistics (cached under DATA/l_swirl/real).
metrics::ClipStats real_stats(const Ctx& c, sim::Effect e, const Setting& s, std::uint64_t seed) {
  const fs::path p = c.dir() / "real" / std::format("{}_{}.stats", ename(e), seed);
  metrics::ClipStats st;
  if (load_stats(p, st)) return st;
  sim::Fluid f(setting_params(e, s, seed));
  for (int i = 0; i < warm_of(e); ++i) f.step_frame();
  Clip cl;
  cl.allocate(kSize, frames_of(e));
  cl.fps = 30.f;
  for (int i = 0; i < cl.frames; ++i) {
    f.step_frame();
    f.render(cl.frame(i));
  }
  st = metrics::stats(cl);
  save_stats(p, st);
  return st;
}
// Training settings: the controls of the effect's salt-1 recipe runs (the runs the stepper was trained on).
std::vector<Setting> training_settings(sim::Effect e, int n, int first = 0) {
  std::vector<Setting> v;
  for (int i = first; i < first + n; ++i) {
    const sim::Params p = run_params(e, 1, static_cast<std::uint64_t>(i));
    v.push_back({p.intensity, p.wind, p.turbulence});
  }
  return v;
}

// --- dstats: the endless statistics, taken apart (training settings) -----------------------------------------------------

// Per training setting: a real run A (the reference) and a real run B; B's own frames are the floor, B's true fields drawn
// by the learned renderer show the renderer's share, the detail layer on B's coarse states (drawn by the simulation's
// renderer and by the learned one) the detail layer's, and the model's own play (runtime, shards; and one continuous
// reference rollout drawn both ways) the stepper's.
void step_dstats(const Ctx& c, sim::Effect e, const tools::Args& a) {
  const rollout::Model M = load_or_throw(a.need("model"));
  const int n_set = a.i("settings", 8), first = a.i("first", 0);
  const auto settings = training_settings(e, n_set, first);
  std::vector<Variant> variants = {make_variant(M, "current")};
  for (const auto& v : split_list(a.str("variants", ""), '/')) variants.push_back(make_variant(M, v));
  const std::uint64_t base = 5'100'000, play = 5'200'000;
  const int F = frames_of(e), W = warm_of(e);
  std::vector<std::string> methods = {"real_B", "renderer_true", "neural", "neural_cont", "neural_cont_simrender"};
  for (const auto& v : variants) {
    methods.push_back("oracle_detail_" + v.name);
    methods.push_back("oracle_detail_lr_" + v.name);
    if (v.name != "current" && v.xo.plain) methods.push_back("neural_" + v.name);  // the runtime with the variant's constants
    if (v.name != "current" && !v.xo.plain) methods.push_back("neural_cont_" + v.name);  // the reference rollout with the variant
  }
  std::vector<std::vector<metrics::StatDistance>> d(methods.size(), std::vector<metrics::StatDistance>(sz(n_set)));
  const auto t0 = std::chrono::steady_clock::now();
  parallel(c.threads, n_set, [&](int si) {
    const Setting& s = settings[sz(si)];
    const std::vector<float> ctl{s[0], s[1], s[2]};
    const metrics::ClipStats ref = real_stats(c, e, s, base + 100 * static_cast<std::uint64_t>(si));
    const auto idx = [&](const std::string& n) { return static_cast<std::size_t>(std::ranges::find(methods, n) - methods.begin()); };
    const sim::Params pb = setting_params(e, s, base + 50 + 100 * static_cast<std::uint64_t>(si));
    sim::Fluid fb(pb);
    for (int i = 0; i < W; ++i) fb.step_frame();
    const sim::State sb0 = fb.state();
    Clip real_b, rend;
    real_b.allocate(kSize, F);
    rend.allocate(kSize, F);
    std::vector<std::vector<float>> coarse(sz(F));
    std::vector<float> rgba(sz(kSize) * kSize * 4);
    const int C = M.h.channels();
    for (int i = 0; i < F; ++i) {
      fb.step_frame();
      fb.render(real_b.frame(i));
      const sim::State st = fb.state();
      coarse[sz(i)].resize(sz(kRes) * kRes * rollout::kPhys);
      rollout::coarse_from_sim(st, kRes, pb.fps, coarse[sz(i)]);
      rollout::State rs;
      rs.res = kRes;
      rs.size = kSize;
      rs.coarse.assign(sz(kRes) * kRes * sz(C), 0.f);
      for (int q = 0; q < kRes * kRes; ++q) {
        for (int k = 0; k < rollout::kPhys; ++k) rs.coarse[sz(q) * sz(C) + sz(k)] = coarse[sz(i)][sz(q) * rollout::kPhys + sz(k)];
      }
      rs.fine_t = st.temp;
      rs.fine_d = st.soot;
      rollout::render(M, rs, rgba);
      auto o = rend.frame(i);
      for (std::size_t q = 0; q < rgba.size(); ++q) o[q] = static_cast<std::uint8_t>(rgba[q] * 255.f + 0.5f);
    }
    d[idx("real_B")][sz(si)] = metrics::distance(ref, metrics::stats(real_b));
    d[idx("renderer_true")][sz(si)] = metrics::distance(ref, metrics::stats(rend));
    for (const auto& v : variants) {
      rollout::State rs;
      rs.res = kRes;
      rs.size = kSize;
      rs.time = sb0.time;
      const int Cv = v.m.h.channels();
      rs.coarse.assign(sz(kRes) * kRes * sz(Cv), 0.f);
      rs.flow.assign(sz(kRes) * kRes * 2, 0.f);
      rs.pressure.assign(sz(kRes) * kRes, 0.f);
      rs.fine_t = sb0.temp;
      rs.fine_d = sb0.soot;
      rs.since_start = 1.f;
      Clip od, odl;
      od.allocate(kSize, F);
      odl.allocate(kSize, F);
      for (int i = 0; i < F; ++i) {
        for (int j = 0; j < kRes * kRes; ++j) {
          for (int q = 0; q < rollout::kPhys; ++q) rs.coarse[sz(j) * sz(Cv) + sz(q)] = coarse[sz(i)][sz(j) * rollout::kPhys + sz(q)];
          rs.flow[sz(j) * 2] = coarse[sz(i)][sz(j) * rollout::kPhys];
          rs.flow[sz(j) * 2 + 1] = coarse[sz(i)][sz(j) * rollout::kPhys + 1];
        }
        detail_step_x(v.m, rs, pb.seed, ctl, v.xo);
        rs.time += 1.f / v.m.fps;
        rs.since_start += 1.f / v.m.fps;
        render_fields(pb, rs.fine_t, rs.fine_d, od.frame(i));
        rollout::render(v.m, rs, rgba);
        auto o = odl.frame(i);
        for (std::size_t q = 0; q < rgba.size(); ++q) o[q] = static_cast<std::uint8_t>(rgba[q] * 255.f + 0.5f);
      }
      d[idx("oracle_detail_" + v.name)][sz(si)] = metrics::distance(ref, metrics::stats(od));
      d[idx("oracle_detail_lr_" + v.name)][sz(si)] = metrics::distance(ref, metrics::stats(odl));
    }
    {
      Runtime rt(M);
      d[idx("neural")][sz(si)] = metrics::distance(ref, metrics::stats(runtime_clip(rt.e, ctl, -1, play + 100 * static_cast<std::uint64_t>(si), F)));
    }
    for (const auto& v : variants) {
      if (v.name == "current" || !v.xo.plain) continue;
      Runtime rt(v.m);
      d[idx("neural_" + v.name)][sz(si)] = metrics::distance(ref, metrics::stats(runtime_clip(rt.e, ctl, -1, play + 100 * static_cast<std::uint64_t>(si), F)));
    }
    {  // one continuous reference rollout from the nearest start point, drawn by the learned and by the simulation's renderer
      int best = 0;
      float bd = 1e30f;
      for (std::size_t k = 0; k < M.starts.size(); ++k) {
        float dd = 0;
        for (std::size_t q = 0; q < 3; ++q) dd += (M.starts[k].controls[q] - ctl[q]) * (M.starts[k].controls[q] - ctl[q]);
        if (dd < bd) {
          bd = dd;
          best = static_cast<int>(k);
        }
      }
      const std::uint64_t seed = play + 50 + 100 * static_cast<std::uint64_t>(si);
      rollout::State rs = rollout::start(M, best, kSize, ctl, seed);
      Clip nl, ns;
      nl.allocate(kSize, F);
      ns.allocate(kSize, F);
      for (int i = 0; i < F; ++i) {
        rollout::step(M, rs, ctl, seed);
        rollout::render(M, rs, rgba);
        auto o = nl.frame(i);
        for (std::size_t q = 0; q < rgba.size(); ++q) o[q] = static_cast<std::uint8_t>(rgba[q] * 255.f + 0.5f);
        render_fields(pb, rs.fine_t, rs.fine_d, ns.frame(i));
      }
      d[idx("neural_cont")][sz(si)] = metrics::distance(ref, metrics::stats(nl));
      d[idx("neural_cont_simrender")][sz(si)] = metrics::distance(ref, metrics::stats(ns));
      for (const auto& v : variants) {  // the same rollout (start, seed) with a variant's detail layer, learned renderer
        if (v.name == "current" || v.xo.plain) continue;
        rollout::State vs = rollout::start(v.m, best, kSize, ctl, seed);
        Clip vc;
        vc.allocate(kSize, F);
        for (int i = 0; i < F; ++i) {
          std::vector<float> cond(sz(v.m.h.cond())), noise(sz(kRes) * kRes * rollout::kNoise), next(vs.coarse.size());
          rollout::condition(v.m, ctl, vs.time, cond);
          rollout::coarse_noise(v.m, seed, vs.time, noise);
          rollout::coarse_step(v.m, vs.coarse, noise, cond, vs.pressure, next, vs.flow);
          vs.coarse.swap(next);
          detail_step_x(v.m, vs, seed, ctl, v.xo);
          vs.time += 1.f / v.m.fps;
          vs.since_start += 1.f / v.m.fps;
          rollout::render(v.m, vs, rgba);
          auto o = vc.frame(i);
          for (std::size_t q = 0; q < rgba.size(); ++q) o[q] = static_cast<std::uint8_t>(rgba[q] * 255.f + 0.5f);
        }
        d[idx("neural_cont_" + v.name)][sz(si)] = metrics::distance(ref, metrics::stats(vc));
      }
    }
    std::println("dstats: {} setting {} done ({:.0f} s)", ename(e), si, seconds_since(t0));
    std::fflush(stdout);
  });
  std::println("\n{}: endless statistics against a real run, mean over {} training settings", ename(e), n_set);
  std::println("{:<30}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}", "method", "spectrum", "|ln mot|", "motion", "coverage", "emission", "PSNR");
  std::vector<std::string> rows;
  for (std::size_t m = 0; m < methods.size(); ++m) {
    double sp = 0, lm = 0, mo = 0, cv = 0, em = 0, ps = 0;
    for (int si = 0; si < n_set; ++si) {
      const auto& x = d[m][sz(si)];
      sp += x.spectrum_l1;
      lm += std::abs(std::log(std::max(1e-3, x.motion_ratio)));
      mo += x.motion_ratio;
      cv += x.coverage_l1;
      em += x.emission_l1;
      ps += x.mean_frame_psnr;
      rows.push_back(std::format("{},{},{},{:.5f},{:.5f},{:.6f},{:.6f},{:.4f}", ename(e), methods[m], si, x.spectrum_l1, x.motion_ratio, x.coverage_l1, x.emission_l1,
                                 x.mean_frame_psnr));
    }
    const double n = n_set;
    std::println("{:<30}{:>10.4f}{:>10.4f}{:>10.3f}{:>10.5f}{:>10.5f}{:>10.2f}", methods[m], sp / n, lm / n, mo / n, cv / n, em / n, ps / n);
  }
  write_rows(c.dir() / "diag" / std::format("{}_{}.csv", ename(e), a.str("tag", "dstats")),
             "effect,method,setting,spectrum_l1,motion_ratio,coverage_l1,emission_l1,mean_frame_psnr", rows);
}

// --- column: where the vortex loses its column (training settings) ------------------------------------------------------

double cover_of(std::span<const std::uint8_t> f) {  // the share of pixels with alpha above 16 (as the check after the test)
  std::size_t n = 0;
  for (std::size_t i = 3; i < f.size(); i += 4) n += f[i] > 16 ? 1 : 0;
  return static_cast<double>(n) / static_cast<double>(f.size() / 4);
}
struct CoverRun {
  double mean = 0, below = 0, longest = 0;  // mean cover; share of frames below a quarter of the real mean; longest stretch (s)
  int first_loss = -1;                      // first frame below a quarter (-1: none)
};
CoverRun cover_run(const Clip& clip, double real_mean) {
  CoverRun r;
  int run = 0, longest = 0, below = 0;
  for (int i = 0; i < clip.frames; ++i) {
    const double cv = cover_of(clip.frame(i));
    r.mean += cv / clip.frames;
    if (cv < 0.25 * real_mean) {
      ++below;
      ++run;
      longest = std::max(longest, run);
      if (r.first_loss < 0) r.first_loss = i;
    } else {
      run = 0;
    }
  }
  r.below = static_cast<double>(below) / clip.frames;
  r.longest = longest / 30.0;
  return r;
}
double real_cover(const sim::Params& p, int warm, int frames) {
  sim::Fluid f(p);
  for (int i = 0; i < warm; ++i) f.step_frame();
  std::vector<std::uint8_t> fr(sz(kSize) * kSize * 4);
  double s = 0;
  for (int i = 0; i < frames; ++i) {
    f.step_frame();
    f.render(fr);
    s += cover_of(fr) / frames;
  }
  return s;
}

// A sheet of one low-intensity training run (vortex): the real run, shards, its own true state, each start point alone;
// every 2 s for 20 s.
void step_column_figure(const Ctx& c, sim::Effect e, const tools::Args& a) {
  (void)c;
  const rollout::Model M = load_or_throw(a.need("model"));
  const int run = a.i("run", 4);
  const sim::Params p = run_params(e, 1, static_cast<std::uint64_t>(run));
  const std::vector<float> ctl{p.intensity, p.wind, p.turbulence};
  const int F = 600;
  std::vector<int> cols;
  for (int f = 59; f < F; f += 60) cols.push_back(f);
  const auto pick = [&](const Clip& src) {
    Clip o;
    o.allocate(kSize, static_cast<int>(cols.size()));
    for (std::size_t q = 0; q < cols.size(); ++q) std::ranges::copy(src.frame(cols[q]), o.frame(static_cast<int>(q)).begin());
    return o;
  };
  std::vector<Clip> rows;
  {
    sim::Params q = p;
    q.seed = 5'160'000 + static_cast<std::uint64_t>(run);
    sim::Fluid f(q);
    for (int i = 0; i < 150; ++i) f.step_frame();
    Clip real;
    real.allocate(kSize, F);
    for (int i = 0; i < F; ++i) {
      f.step_frame();
      f.render(real.frame(i));
    }
    rows.push_back(pick(real));
  }
  {
    Runtime rt(M);
    rows.push_back(pick(runtime_clip(rt.e, ctl, -1, 5'160'050 + static_cast<std::uint64_t>(run), F)));
  }
  {
    sim::Fluid f(p);
    for (int i = 0; i < 100; ++i) f.step_frame();
    const sim::State st = f.state();
    rollout::Model m1 = M;
    rollout::StartPoint sp;
    sp.controls = ctl;
    sp.seed = p.seed;
    sp.time = st.time;
    sp.coarse.resize(sz(kRes) * kRes * rollout::kPhys);
    rollout::coarse_from_sim(st, kRes, p.fps, sp.coarse);
    m1.starts = {sp};
    Runtime r1(m1);
    rows.push_back(pick(runtime_clip(r1.e, ctl, 0, 0, F, false)));
  }
  for (std::size_t k = 0; k < M.starts.size(); ++k) {
    rollout::Model m1 = M;
    m1.starts = {M.starts[k]};
    m1.starts[0].seed = 5'160'060 + k;
    Runtime r1(m1);
    rows.push_back(pick(runtime_clip(r1.e, ctl, 0, 0, F, false)));
  }
  std::vector<const Clip*> rp;
  for (const auto& r : rows) rp.push_back(&r);
  if (auto w = write_png(a.need("out"), comparison_sheet(rp, static_cast<int>(cols.size()), Background::black, 1)); !w) throw std::runtime_error(w.error());
}

// Training coverage at low intensity, the start points, and three ways to play at low-intensity training settings: as
// shipped (shards), from each start point alone, and from the run's own true state (a perfect start point).
void step_column(const Ctx& c, sim::Effect e, const tools::Args& a) {
  const rollout::Model M = load_or_throw(a.need("model"));
  const float imax = a.f("max-intensity", 0.4f);
  const int want = a.i("settings", 8), seconds = a.i("seconds", 60);
  rollout::SimRecipe r = rollout::recipe_for(e);
  std::vector<int> hist(10, 0), low;
  for (int i = 0; i < r.runs; ++i) {
    const sim::Params p = run_params(e, 1, static_cast<std::uint64_t>(i));
    ++hist[sz(std::min(9, static_cast<int>(p.intensity * 10.f)))];
    if (p.intensity <= imax && static_cast<int>(low.size()) < want) low.push_back(i);
  }
  std::string h;
  for (int b = 0; b < 10; ++b) h += std::format(" {:.1f}-{:.1f}: {}", b / 10.0, (b + 1) / 10.0, hist[sz(b)]);
  std::println("training runs by intensity:{}", h);
  std::println("start points (controls; coarse heat in the bottom 4 rows, middle 8 columns):");
  for (std::size_t k = 0; k < M.starts.size(); ++k) {
    const auto& sp = M.starts[k];
    double foot = 0;
    for (int y = 0; y < 4; ++y) {
      for (int x = 12; x < 20; ++x) foot += sp.coarse[(sz(y) * kRes + sz(x)) * rollout::kPhys + 2];
    }
    std::println("  {}: ({:.2f}, {:.2f}, {:.2f}) foot heat {:.3f}", k, sp.controls[0], sp.controls[1], sp.controls[2], foot);
  }
  std::vector<std::string> rows;
  const std::uint64_t base = 5'150'000;
  parallel(c.threads, static_cast<int>(low.size()), [&](int li) {
    const int run = low[sz(li)];
    const sim::Params p = run_params(e, 1, static_cast<std::uint64_t>(run));
    const std::vector<float> ctl{p.intensity, p.wind, p.turbulence};
    sim::Params q = p;
    q.seed = base + 100 * static_cast<std::uint64_t>(li);
    const double rc = real_cover(q, 150, 300);
    Runtime rt(M);
    std::vector<std::string> out;
    const CoverRun sh = cover_run(runtime_clip(rt.e, ctl, -1, base + 50 + 100 * static_cast<std::uint64_t>(li), seconds * 30), rc);
    out.push_back(std::format("{},{},{:.2f},{:.2f},{:.2f},shards,-1,{:.3f},{:.3f},{:.1f},{:.1f},{}", ename(e), run, ctl[0], ctl[1], ctl[2], rc, sh.mean, 100 * sh.below,
                              sh.longest, sh.first_loss));
    for (std::size_t k = 0; k < M.starts.size(); ++k) {  // each start point alone, one continuous rollout, a new seed
      rollout::Model m1 = M;
      m1.starts = {M.starts[k]};
      m1.starts[0].seed = base + 60 + 100 * static_cast<std::uint64_t>(li) + k;  // played with a seed of its own
      Runtime r1(m1);
      const CoverRun cr = cover_run(runtime_clip(r1.e, ctl, 0, 0, 20 * 30, false), rc);
      out.push_back(std::format("{},{},{:.2f},{:.2f},{:.2f},start{},{},{:.3f},{:.3f},{:.1f},{:.1f},{}", ename(e), run, ctl[0], ctl[1], ctl[2], k, k, rc, cr.mean,
                                100 * cr.below, cr.longest, cr.first_loss));
    }
    {  // an empty grid as the only start point (as every training run begins), the run's own seed: does the column grow?
      rollout::Model m1 = M;
      rollout::StartPoint sp;
      sp.controls = ctl;
      sp.seed = p.seed;
      sp.time = 0.f;
      sp.coarse.assign(sz(kRes) * kRes * rollout::kPhys, 0.f);
      m1.starts = {sp};
      Runtime r1(m1);
      const Clip cl = runtime_clip(r1.e, ctl, 0, 0, 20 * 30, false);
      const CoverRun cr = cover_run(cl, rc);
      int grown = -1;  // the first frame from which the cover stays above half the real mean for a second
      for (int i = 0; i + 30 <= cl.frames && grown < 0; ++i) {
        bool ok = true;
        for (int j = i; j < i + 30 && ok; ++j) ok = cover_of(cl.frame(j)) >= 0.5 * rc;
        if (ok) grown = i;
      }
      out.push_back(std::format("{},{},{:.2f},{:.2f},{:.2f},empty,{},{:.3f},{:.3f},{:.1f},{:.1f},{}", ename(e), run, ctl[0], ctl[1], ctl[2], grown, rc, cr.mean,
                                100 * cr.below, cr.longest, cr.first_loss));
      sim::Fluid f(p);  // and the simulation's own growth from empty, with the same seed
      std::vector<std::uint8_t> fr(sz(kSize) * kSize * 4);
      std::vector<double> covers;
      for (int i = 0; i < 20 * 30; ++i) {
        f.step_frame();
        f.render(fr);
        covers.push_back(cover_of(fr));
      }
      int sgrown = -1;
      for (int i = 0; i + 30 <= static_cast<int>(covers.size()) && sgrown < 0; ++i) {
        bool ok = true;
        for (int j = i; j < i + 30 && ok; ++j) ok = covers[sz(j)] >= 0.5 * rc;
        if (ok) sgrown = i;
      }
      out.push_back(std::format("{},{},{:.2f},{:.2f},{:.2f},empty_simulation,{},{:.3f},,,,", ename(e), run, ctl[0], ctl[1], ctl[2], sgrown, rc));
    }
    {  // the run's own state at frame 100 as the only start point (coarse only, as the stored ones), its own seed
      sim::Fluid f(p);
      for (int i = 0; i < 100; ++i) f.step_frame();
      const sim::State st = f.state();
      rollout::Model m1 = M;
      rollout::StartPoint sp;
      sp.controls = ctl;
      sp.seed = p.seed;
      sp.time = st.time;
      sp.coarse.resize(sz(kRes) * kRes * rollout::kPhys);
      rollout::coarse_from_sim(st, kRes, p.fps, sp.coarse);
      m1.starts = {sp};
      Runtime r1(m1);
      const CoverRun cr = cover_run(runtime_clip(r1.e, ctl, 0, 0, 20 * 30, false), rc);
      out.push_back(std::format("{},{},{:.2f},{:.2f},{:.2f},own_state,-1,{:.3f},{:.3f},{:.1f},{:.1f},{}", ename(e), run, ctl[0], ctl[1], ctl[2], rc, cr.mean, 100 * cr.below,
                                cr.longest, cr.first_loss));
    }
    for (const auto& o : out) std::println("{}", o);
    std::fflush(stdout);
    static std::mutex mu;
    std::lock_guard lk(mu);
    rows.insert(rows.end(), out.begin(), out.end());
  });
  write_rows(c.dir() / "diag" / std::format("{}_{}.csv", ename(e), a.str("tag", "column")),
             "effect,run,intensity,second,turbulence,play,start,real_cover,cover,pct_below_quarter,longest_s,first_loss_frame", rows);
}


// --- validation and test: the current model against candidates -------------------------------------------------------------

bool off_grid(const Setting& s) {
  const auto off = [](float x, std::initializer_list<float> g) { return std::ranges::all_of(g, [x](float y) { return std::abs(x - y) >= 0.05f; }); };
  return off(s[0], {0.f, 0.5f, 1.f}) && off(s[1], {0.f, 0.25f, 0.5f, 0.75f, 1.f}) && off(s[2], {0.f, 0.5f, 1.f});
}
// Study B's ten held-out settings (the test) and study G's ten validation settings (docs/DCM.md §4), as nvfx_d2.
std::vector<Setting> test_settings() {
  std::mt19937_64 rng(2026);
  std::uniform_real_distribution<float> u(0.05f, 0.95f);
  std::vector<Setting> v;
  while (v.size() < 10) {
    const Setting s{u(rng), u(rng), u(rng)};
    if (off_grid(s)) v.push_back(s);
  }
  return v;
}
std::vector<Setting> validation_settings() {
  const auto test = test_settings();
  std::mt19937_64 rng(2027);
  std::uniform_real_distribution<float> u(0.05f, 0.95f);
  std::vector<Setting> v;
  while (v.size() < 10) {
    const Setting s{u(rng), u(rng), u(rng)};
    const bool far = std::ranges::all_of(test, [&](const Setting& t) {
      return std::sqrt((s[0] - t[0]) * (s[0] - t[0]) + (s[1] - t[1]) * (s[1] - t[1]) + (s[2] - t[2]) * (s[2] - t[2])) >= 0.05f;
    });
    if (off_grid(s) && far) v.push_back(s);
  }
  return v;
}

// Seeds of study L (docs/REPORT.md §12, study L; no earlier study used these bases): per split, real run k of setting s
// has seed real + 100 s + k (k < 8), the floor's runs real + 50 + 100 s + k, the models' plays model + 100 s + k, and the
// tracking runs track + 100 s + k (k < 4), each simulated from an empty grid for 100 frames and followed from its true
// state for 240 frames. Validation tracks the studies' salt-3 runs 0 to 15 instead (their own settings and seeds).
struct Split {
  std::string name;
  std::vector<Setting> settings;
  int reps = 8, track_reps = 4;
  std::uint64_t real = 0, model = 0, track = 0;
  int track_salt = 0, track_runs = 0;  // > 0: the tracking runs are recipe runs 0 .. track_runs - 1 of this salt instead
};
Split split_of(const std::string& name, sim::Effect e) {
  Split s;
  s.name = name;
  if (name == "probe") {  // development only: the controls of training runs 0 to 9, fewer runs
    s.settings = training_settings(e, 10);
    s.reps = 4;
    s.track_reps = 2;
    s.real = 5'100'000;
    s.model = 5'200'000;
    s.track = 5'250'000;
  } else if (name == "val") {
    s.settings = validation_settings();
    s.real = 5'300'000;
    s.model = 5'400'000;
    s.track_salt = 3;  // the studies' validation runs (salt 3), each its own setting and seed
    s.track_runs = 16;
  } else if (name == "test") {
    s.settings = test_settings();
    s.real = 5'700'000;
    s.model = 5'800'000;
    s.track = 5'900'000;
  } else {
    throw std::invalid_argument("unknown split " + name);
  }
  return s;
}

metrics::ClipStats pool(const std::vector<metrics::ClipStats>& v) {  // as nvfx_d2: per-frame curves, spectrum, mean frame, motion averaged
  metrics::ClipStats p = v.at(0);
  const double n = static_cast<double>(v.size());
  for (std::size_t k = 1; k < v.size(); ++k) {
    for (std::size_t i = 0; i < p.coverage.size(); ++i) p.coverage[i] += v[k].coverage[i];
    for (std::size_t i = 0; i < p.emission.size(); ++i) p.emission[i] += v[k].emission[i];
    for (std::size_t i = 0; i < p.spectrum.size(); ++i) p.spectrum[i] += v[k].spectrum[i];
    for (std::size_t i = 0; i < p.mean_frame.size(); ++i) p.mean_frame[i] += v[k].mean_frame[i];
    p.motion += v[k].motion;
  }
  for (double& x : p.coverage) x /= n;
  for (double& x : p.emission) x /= n;
  for (double& x : p.spectrum) x /= n;
  for (float& x : p.mean_frame) x /= static_cast<float>(n);
  p.motion /= n;
  return p;
}

struct Entry {
  std::string name;
  fs::path path;
  rollout::Model m;
  std::string sha;
};
std::string file_sha256(const fs::path& p);  // below

constexpr std::array kTrackH = {1, 8, 30, 60, 240};
constexpr int kTrack1s = 2;  // index of 30 frames (1 s) in kTrackH
const std::array<std::string, 5> kStatNames = {"spectrum_l1", "abs_ln_motion_ratio", "coverage_l1", "emission_l1", "mean_frame_psnr"};

struct Scores {
  std::vector<std::array<double, kTrackH.size()>> track;      // per setting: mean over its tracking runs
  std::vector<std::array<double, kTrackH.size()>> track_csd;  // the same for the coarse simulation with the model's detail layer
  std::vector<metrics::StatDistance> d, d1;                   // per setting: pooled plays against pooled real; first against first
};
double stat_value(const metrics::StatDistance& x, int k) {
  switch (k) {
    case 0: return x.spectrum_l1;
    case 1: return std::abs(std::log(std::max(1e-3, x.motion_ratio)));
    case 2: return x.coverage_l1;
    case 3: return x.emission_l1;
    default: return x.mean_frame_psnr;
  }
}

// Tracking (the runtime from the true state, as d-eval) and the endless statistics (the runtime as shipped: shards from
// start points, pooled over runs) of every entry on a split. Real runs' statistics are cached.
std::vector<Scores> evaluate(const Ctx& c, sim::Effect e, const Split& sp, const std::vector<Entry>& models, bool with_csd) {
  const int n = static_cast<int>(sp.settings.size()), KE = 240;
  const int nt = sp.track_salt > 0 ? sp.track_runs : n, per = sp.track_salt > 0 ? 1 : sp.track_reps;  // tracking units, runs each
  std::vector<Scores> out(models.size());
  for (auto& s : out) {
    s.track.assign(sz(nt), {});
    s.track_csd.assign(sz(nt), {});
    s.d.resize(sz(n));
    s.d1.resize(sz(n));
  }
  const auto t0 = std::chrono::steady_clock::now();
  // tracking: per (setting, run) the truth once, every model on it
  std::vector<std::vector<std::vector<std::array<double, kTrackH.size()>>>> tr(models.size(), std::vector<std::vector<std::array<double, kTrackH.size()>>>(sz(nt))),
      trc = tr;
  std::mutex mu;
  parallel(c.threads, nt * per, [&](int j) {
    const int si = j / per, k = j % per;
    const sim::Params p = sp.track_salt > 0 ? run_params(e, static_cast<std::uint64_t>(sp.track_salt), static_cast<std::uint64_t>(si))
                                            : setting_params(e, sp.settings[sz(si)], sp.track + 100 * static_cast<std::uint64_t>(si) + static_cast<std::uint64_t>(k));
    const std::vector<float> ctl{p.intensity, p.wind, p.turbulence};
    sim::Fluid truth(p);
    for (int i = 0; i < 100; ++i) truth.step_frame();
    const sim::State st = truth.state();
    Clip ref;
    ref.allocate(kSize, KE);
    for (int i = 0; i < KE; ++i) {
      truth.step_frame();
      truth.render(ref.frame(i));
    }
    rollout::StartPoint s0;
    s0.controls = ctl;
    s0.seed = p.seed;
    s0.time = st.time;
    s0.coarse.resize(sz(kRes) * kRes * rollout::kPhys);
    rollout::coarse_from_sim(st, kRes, p.fps, s0.coarse);
    s0.fine_t = st.temp;
    s0.fine_d = st.soot;
    for (std::size_t mi = 0; mi < models.size(); ++mi) {
      rollout::Model mt = models[mi].m;
      mt.h.start_fine = kSize;
      mt.starts = {s0};
      Runtime fx(mt);
      const Clip nc = runtime_clip(fx.e, ctl, 0, 0, KE + 1, false);
      std::array<double, kTrackH.size()> a{}, b{};
      for (std::size_t h = 0; h < kTrackH.size(); ++h) a[h] = apsnr(ref.frame(kTrackH[h] - 1), nc.frame(kTrackH[h]));
      if (with_csd) {  // the coarse simulation on the same 32-cell grid with the model's detail layer (d-eval's coarse_sim_detail)
        sim::Params q = p;
        q.sim_res = kRes;
        sim::Fluid f(q);
        sim::State cs;
        cs.n = kRes;
        cs.frame = st.frame;
        cs.time = st.time;
        const auto n2 = sz(kRes) * kRes;
        for (auto* v : {&cs.u, &cs.v, &cs.temp, &cs.soot, &cs.pressure}) v->assign(n2, 0.f);
        const int kb = st.n / kRes;
        const float ia = 1.f / fl(kb * kb);
        for (int y = 0; y < st.n; ++y) {
          for (int x = 0; x < st.n; ++x) {
            const std::size_t i = sz(y) * sz(st.n) + sz(x), o = sz(y / kb) * kRes + sz(x / kb);
            cs.u[o] += st.u[i] * ia / fl(kb);
            cs.v[o] += st.v[i] * ia / fl(kb);
            cs.temp[o] += st.temp[i] * ia;
            cs.soot[o] += st.soot[i] * ia;
          }
        }
        f.set_state(cs);
        const rollout::Model& M = models[mi].m;
        rollout::State rs;
        rs.res = kRes;
        rs.size = kSize;
        rs.time = st.time;
        const int C = M.h.channels();
        rs.coarse.assign(n2 * sz(C), 0.f);
        rs.flow.assign(n2 * 2, 0.f);
        rs.pressure.assign(n2, 0.f);
        rs.fine_t = st.temp;
        rs.fine_d = st.soot;
        std::vector<std::uint8_t> fr(sz(kSize) * kSize * 4);
        std::size_t h = 0;
        for (int i = 0; i < KE; ++i) {
          f.step_frame();
          const sim::State x = f.state();
          for (std::size_t q2 = 0; q2 < n2; ++q2) {
            rs.flow[q2 * 2] = x.u[q2] / p.fps;
            rs.flow[q2 * 2 + 1] = x.v[q2] / p.fps;
            rs.coarse[q2 * sz(C) + 2] = x.temp[q2];
            rs.coarse[q2 * sz(C) + 3] = x.soot[q2];
          }
          rollout::detail_step(M, rs, p.seed, ctl);
          rs.time += 1.f / p.fps;
          rs.since_start += 1.f / p.fps;
          if (h < kTrackH.size() && i + 1 == kTrackH[h]) {
            render_fields(p, rs.fine_t, rs.fine_d, fr);
            b[h++] = apsnr(ref.frame(i), fr);
          }
        }
      }
      std::lock_guard lk(mu);
      tr[mi][sz(si)].push_back(a);
      trc[mi][sz(si)].push_back(b);
    }
    std::println("{}: {} tracking setting {} run {} done ({:.0f} s)", sp.name, ename(e), si, k, seconds_since(t0));
    std::fflush(stdout);
  });
  for (std::size_t mi = 0; mi < models.size(); ++mi) {
    for (int si = 0; si < nt; ++si) {
      for (std::size_t h = 0; h < kTrackH.size(); ++h) {
        double s1 = 0, s2 = 0;
        for (const auto& v : tr[mi][sz(si)]) s1 += v[h] / static_cast<double>(tr[mi][sz(si)].size());
        for (const auto& v : trc[mi][sz(si)]) s2 += v[h] / static_cast<double>(trc[mi][sz(si)].size());
        out[mi].track[sz(si)][h] = s1;
        out[mi].track_csd[sz(si)][h] = s2;
      }
    }
  }
  // endless: real runs (cached), then the plays of every model
  parallel(c.threads, n * sp.reps * 2, [&](int j) {
    const int si = j % n, k = (j / n) % sp.reps, o = j / (n * sp.reps);
    (void)real_stats(c, e, sp.settings[sz(si)], sp.real + (o ? 50 : 0) + 100 * static_cast<std::uint64_t>(si) + static_cast<std::uint64_t>(k));
  });
  std::println("{}: {} real runs ready ({:.0f} s)", sp.name, ename(e), seconds_since(t0));
  std::vector<metrics::ClipStats> ref(sz(n)), other(sz(n)), ref1(sz(n));
  for (int si = 0; si < n; ++si) {
    for (int o = 0; o < 2; ++o) {
      std::vector<metrics::ClipStats> v;
      for (int k = 0; k < sp.reps; ++k) {
        v.push_back(real_stats(c, e, sp.settings[sz(si)], sp.real + (o ? 50 : 0) + 100 * static_cast<std::uint64_t>(si) + static_cast<std::uint64_t>(k)));
      }
      (o ? other : ref)[sz(si)] = pool(v);
      if (!o) ref1[sz(si)] = v[0];
    }
  }
  for (std::size_t mi = 0; mi < models.size(); ++mi) {
    Runtime rt(models[mi].m);
    std::vector<metrics::ClipStats> runs(sz(n * sp.reps));
    parallel(c.threads, n * sp.reps, [&](int j) {
      const int si = j / sp.reps, k = j % sp.reps;
      const std::vector<float> ctl(sp.settings[sz(si)].begin(), sp.settings[sz(si)].end());
      runs[sz(j)] = metrics::stats(runtime_clip(rt.e, ctl, -1, sp.model + 100 * static_cast<std::uint64_t>(si) + static_cast<std::uint64_t>(k), frames_of(e)));
    });
    for (int si = 0; si < n; ++si) {
      const metrics::ClipStats pl = pool(std::vector<metrics::ClipStats>(runs.begin() + si * sp.reps, runs.begin() + (si + 1) * sp.reps));
      out[mi].d[sz(si)] = metrics::distance(ref[sz(si)], pl);
      out[mi].d1[sz(si)] = metrics::distance(ref1[sz(si)], runs[sz(si * sp.reps)]);
    }
  }
  Scores floor;
  for (int si = 0; si < n; ++si) floor.d.push_back(metrics::distance(ref[sz(si)], other[sz(si)]));
  out.push_back(floor);  // the last: real, other seeds (the floor; no tracking)
  std::println("{}: {} plays done ({:.0f} s)", sp.name, ename(e), seconds_since(t0));
  return out;
}

// The rule (docs/REPORT.md §12, study L; committed before validation). Candidate minus current model, paired over the 10
// settings, 95% bootstrap with 10,000 resamples (metrics::paired_bootstrap, seed 2026):
//   1. it is better on tracking at 1 s (active PSNR, interval above zero) or on the endless detail spectrum distance
//      (interval below zero);
//   2. no statistic is worse: tracking at 1 s (no interval below zero); spectrum distance, |ln motion ratio|, coverage
//      distance and emission distance (no interval above zero); mean-frame PSNR (no interval below zero);
//   3. the vortex: it loses its column in no larger share of frames than the current model in the column-loss check (60 s
//      at each of the 10 held-out settings; checked by `collapse`, reported beside).
// The statistics, in this order: tracking at 1 s, then the five endless ones (kStatNames).
struct RuleResult {
  std::array<metrics::Interval, 6> iv{};
  bool better = false, none_worse = false;
  bool pass() const { return better && none_worse; }
};
std::vector<double> rule_stat(const Scores& s, int k) {  // tracking: per tracking unit; the rest: per setting
  std::vector<double> v;
  const std::size_t n = k == 0 ? s.track.size() : s.d.size();
  for (std::size_t i = 0; i < n; ++i) v.push_back(k == 0 ? s.track[i][kTrack1s] : stat_value(s.d[i], k - 1));
  return v;
}
RuleResult apply_rule_l(const Scores& cand, const Scores& cur) {
  RuleResult r;
  for (int k = 0; k < 6; ++k) r.iv[sz(k)] = metrics::paired_bootstrap(rule_stat(cand, k), rule_stat(cur, k), 10000, 2026);
  const auto higher_better = [](int k) { return k == 0 || k == 5; };
  const auto better = [&](int k) { return higher_better(k) ? r.iv[sz(k)].lo > 0 : r.iv[sz(k)].hi < 0; };
  const auto worse = [&](int k) { return higher_better(k) ? r.iv[sz(k)].hi < 0 : r.iv[sz(k)].lo > 0; };
  r.better = better(0) || better(1);
  r.none_worse = true;
  for (int k = 0; k < 6; ++k) r.none_worse = r.none_worse && !worse(k);
  return r;
}
const std::array<std::string, 6> kRuleNames = {"track_1s", "spectrum_l1", "abs_ln_motion_ratio", "coverage_l1", "emission_l1", "mean_frame_psnr"};

std::string file_sha256(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  std::stringstream ss;
  ss << f.rdbuf();
  return dcm::sha256_hex(ss.str());
}

void merge_rows(const fs::path& path, const std::string& header, const std::vector<std::string>& rows, const std::string& key) {
  std::vector<std::string> keep;
  if (std::ifstream in(path); in) {
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
      if (!line.empty() && line.substr(0, line.find(',')) != key) keep.push_back(line);
    }
  }
  for (const auto& r : rows) keep.push_back(r);
  write_rows(path, header, keep);
}

// "name=path,name=path": the entries to score; the first is the current model.
std::vector<Entry> read_entries(const std::string& current, const std::string& cands) {
  std::vector<Entry> v;
  v.push_back({"current", current, load_or_throw(current), file_sha256(current)});
  for (const auto& item : split_list(cands, ',')) {
    const auto eq = item.find('=');
    if (eq == std::string::npos) throw std::invalid_argument("--cand: name=path");
    const fs::path p = item.substr(eq + 1);
    v.push_back({item.substr(0, eq), p, load_or_throw(p), file_sha256(p)});
  }
  return v;
}

void report_split(const Ctx& c, sim::Effect e, const Split& sp, const std::vector<Entry>& models, const std::vector<Scores>& sc, const std::string& prefix,
                  bool decide) {
  const std::string en = ename(e);
  std::vector<std::string> track_rows, stat_rows, cmp_rows, choice_rows;
  const std::size_t n = sp.settings.size();
  for (std::size_t mi = 0; mi <= models.size(); ++mi) {
    const bool fl0 = mi == models.size();
    const std::string name = fl0 ? "real_other_seeds" : models[mi].name;
    for (std::size_t si = 0; si < (fl0 ? 0 : sc[mi].track.size()); ++si) {
      std::string row = std::format("{},{},{}", en, name, si);
      for (std::size_t h = 0; h < kTrackH.size(); ++h) row += std::format(",{:.3f}", sc[mi].track[si][h]);
      for (std::size_t h = 0; h < kTrackH.size(); ++h) row += std::format(",{:.3f}", sc[mi].track_csd[si][h]);
      track_rows.push_back(row);
    }
    for (std::size_t si = 0; si < n; ++si) {
      for (int pooled = 1; pooled >= (fl0 ? 1 : 0); --pooled) {
        const auto& x = pooled ? sc[mi].d[si] : sc[mi].d1[si];
        stat_rows.push_back(std::format("{},{},{},{},{:.5f},{:.5f},{:.5f},{:.6f},{:.6f},{:.4f}", en, name, pooled ? std::format("pooled{}", sp.reps) : "one", si,
                                        x.spectrum_l1, x.motion_ratio, std::abs(std::log(std::max(1e-3, x.motion_ratio))), x.coverage_l1, x.emission_l1,
                                        x.mean_frame_psnr));
      }
    }
  }
  // means
  std::println("\n{} {}: means over the tracking runs or settings and the {} settings", sp.name, en, n);
  std::println("{:<22}{:>8}{:>8}{:>8}{:>8}{:>8} |{:>8}{:>8}{:>8} |{:>9}{:>9}{:>9}{:>9}{:>9}{:>8}", "model", "f1", "f8", "1s", "2s", "8s", "csd f1", "csd 1s",
               "csd 8s", "spectrum", "|lnmot|", "motion", "cover", "emiss", "PSNR");
  for (std::size_t mi = 0; mi <= models.size(); ++mi) {
    const bool fl0 = mi == models.size();
    std::array<double, kTrackH.size()> t{}, tc{};
    std::array<double, 6> st{};
    const std::size_t nt = fl0 ? 0 : sc[mi].track.size();
    for (std::size_t si = 0; si < nt; ++si) {
      for (std::size_t h = 0; h < kTrackH.size(); ++h) {
        t[h] += sc[mi].track[si][h] / static_cast<double>(nt);
        tc[h] += sc[mi].track_csd[si][h] / static_cast<double>(nt);
      }
    }
    for (std::size_t si = 0; si < n; ++si) {
      for (int k = 0; k < 5; ++k) st[sz(k)] += stat_value(sc[mi].d[si], k) / static_cast<double>(n);
      st[5] += sc[mi].d[si].motion_ratio / static_cast<double>(n);
    }
    std::println("{:<22}{:>8.2f}{:>8.2f}{:>8.2f}{:>8.2f}{:>8.2f} |{:>8.2f}{:>8.2f}{:>8.2f} |{:>9.4f}{:>9.4f}{:>9.3f}{:>9.5f}{:>9.5f}{:>8.2f}", fl0 ? "real, other seeds" : models[mi].name,
                 t[0], t[1], t[2], t[3], t[4], tc[0], tc[2], tc[4], st[0], st[1], st[5], st[2], st[3], st[4]);
  }
  // each candidate against the current model
  int chosen = -1;
  double best = -1e9;
  for (std::size_t mi = 1; mi < models.size(); ++mi) {
    const RuleResult r = apply_rule_l(sc[mi], sc[0]);
    std::string line = std::format("{} {} {} - current:", sp.name, en, models[mi].name);
    for (int k = 0; k < 6; ++k) {
      const auto& iv = r.iv[sz(k)];
      line += std::format(" {} {:+.4f} [{:+.4f}, {:+.4f}];", kRuleNames[sz(k)], iv.mean, iv.lo, iv.hi);
      const auto a = rule_stat(sc[mi], k), b = rule_stat(sc[0], k);
      cmp_rows.push_back(std::format("{},{},{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}", en, models[mi].name, kRuleNames[sz(k)], iv.mean, iv.lo, iv.hi,
                                     std::accumulate(a.begin(), a.end(), 0.0) / static_cast<double>(a.size()),
                                     std::accumulate(b.begin(), b.end(), 0.0) / static_cast<double>(b.size())));
    }
    for (const int h : {0, 1, 3, 4}) {  // the other tracking horizons, reported
      std::vector<double> a, b;
      for (std::size_t si = 0; si < sc[mi].track.size(); ++si) {
        a.push_back(sc[mi].track[si][sz(h)]);
        b.push_back(sc[0].track[si][sz(h)]);
      }
      const auto iv = metrics::paired_bootstrap(a, b, 10000, 2026);
      const double na = static_cast<double>(a.size());
      cmp_rows.push_back(std::format("{},{},track_f{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}", en, models[mi].name, kTrackH[sz(h)], iv.mean, iv.lo, iv.hi,
                                     std::accumulate(a.begin(), a.end(), 0.0) / na, std::accumulate(b.begin(), b.end(), 0.0) / na));
    }
    std::println("{} -> {}", line, r.pass() ? "MEETS the rule" : "does not meet it");
    choice_rows.push_back(std::format("{},{},{},{},{},{:.4f},{}", en, models[mi].name, r.better ? 1 : 0, r.none_worse ? 1 : 0, r.pass() ? 1 : 0, r.iv[0].mean, models[mi].sha));
    if (r.pass() && r.iv[0].mean > best) {
      best = r.iv[0].mean;
      chosen = static_cast<int>(mi);
    }
  }
  {  // the current model against the coarse simulation with its own detail layer (study D's comparison), for the record
    std::vector<double> a, b;
    for (std::size_t si = 0; si < sc[0].track.size(); ++si) {
      a.push_back(sc[0].track[si][kTrack1s]);
      b.push_back(sc[0].track_csd[si][kTrack1s]);
    }
    const auto iv = metrics::paired_bootstrap(a, b, 10000, 2026);
    std::println("{} {} current - coarse_sim_detail at 1 s: {:+.3f} [{:+.3f}, {:+.3f}]", sp.name, en, iv.mean, iv.lo, iv.hi);
    for (std::size_t mi = 1; mi < models.size(); ++mi) {
      std::vector<double> x, y;
      for (std::size_t si = 0; si < sc[mi].track.size(); ++si) {
        x.push_back(sc[mi].track[si][kTrack1s]);
        y.push_back(sc[mi].track_csd[si][kTrack1s]);
      }
      const auto j = metrics::paired_bootstrap(x, y, 10000, 2026);
      std::println("{} {} {} - its coarse_sim_detail at 1 s: {:+.3f} [{:+.3f}, {:+.3f}]", sp.name, en, models[mi].name, j.mean, j.lo, j.hi);
    }
  }
  std::string th = "effect,model,setting";
  for (const int h : kTrackH) th += std::format(",track_f{}", h);
  for (const int h : kTrackH) th += std::format(",csd_f{}", h);
  merge_rows(c.results / std::format("{}_track.csv", prefix), th, track_rows, en);
  merge_rows(c.results / std::format("{}_stats.csv", prefix), "effect,model,runs,setting,spectrum_l1,motion_ratio,abs_ln_motion_ratio,coverage_l1,emission_l1,mean_frame_psnr",
             stat_rows, en);
  merge_rows(c.results / std::format("{}_compare.csv", prefix), "effect,model,statistic,mean_difference,lo,hi,candidate_mean,current_mean", cmp_rows, en);
  if (decide) {
    choice_rows.push_back(std::format("{},chosen,,,,,{}", en, chosen > 0 ? models[sz(chosen)].name : "none"));
    merge_rows(c.results / std::format("{}_choice.csv", prefix), "effect,model,better,none_worse,meets_rule,track_1s_gain_db,sha256", choice_rows, en);
    std::println("{} {}: chosen {}", sp.name, en, chosen > 0 ? models[sz(chosen)].name : std::string("none (no candidate meets the rule)"));
  }
}

void step_val(const Ctx& c, sim::Effect e, const tools::Args& a) {
  const auto models = read_entries(a.need("current"), a.str("cand", ""));
  const Split sp = split_of(a.str("split", "val"), e);
  for (const auto& m : models) std::println("val: {} {} {} sha256 {}", ename(e), m.name, m.path.string(), m.sha);
  const auto sc = evaluate(c, e, sp, models, !a.flag("no-csd"));
  report_split(c, e, sp, models, sc, a.str("prefix", "l_val"), true);
}

// The test, once: the current model and the frozen candidate (its SHA-256 must match the row in --frozen).
void step_test(const Ctx& c, sim::Effect e, const tools::Args& a) {
  const fs::path done = c.dir() / std::format("test_{}.done", ename(e));
  if (fs::exists(done)) throw std::runtime_error("the test of " + ename(e) + " has run (it runs once)");
  const auto models = read_entries(a.need("current"), a.need("cand"));
  if (models.size() != 2) throw std::invalid_argument("test: one candidate");
  {  // frozen before the test
    std::ifstream in(a.need("frozen"));
    std::string line;
    bool ok = false;
    while (std::getline(in, line)) {
      if (line.find(models[1].sha) != std::string::npos && line.find(ename(e)) != std::string::npos) ok = true;
    }
    if (!ok) throw std::runtime_error("test: the candidate's SHA-256 is not in " + a.need("frozen"));
  }
  std::ofstream(done) << "started " << models[1].sha << "\n";
  const Split sp = split_of("test", e);
  for (const auto& m : models) std::println("test: {} {} {} sha256 {}", ename(e), m.name, m.path.string(), m.sha);
  const auto sc = evaluate(c, e, sp, models, true);
  report_split(c, e, sp, models, sc, a.str("prefix", "l_test"), true);
  std::ofstream(done) << "done " << models[1].sha << "\n";
}

// The column-loss check of docs/EFFECTS_ENERGY.md §4.4, as it was run there: 60 s through the runtime at each of the 10
// held-out settings, and the share of frames whose cover (alpha above 16) falls below a quarter of a real run's mean cover
// (the real run: seed 900000 + k, 150 frames of warm-up, 300 frames). Seeds: --seed-base (950000 as published; this study
// also uses fresh ones).
void step_collapse(const Ctx& c, sim::Effect e, const tools::Args& a) {
  const auto models = read_entries(a.need("current"), a.str("cand", ""));
  const std::uint64_t base = a.u64("seed-base", 950000);
  const int seconds = a.i("seconds", 60);
  // test: study B's held-out settings, the real run of d-eval (seed 900000 + k); val: study G's validation settings, the
  // first real run of this study's validation (seed 5,300,000 + 100 k)
  const bool val = a.str("settings", "test") == "val";
  const auto settings = val ? validation_settings() : test_settings();
  std::vector<double> rc(settings.size());
  parallel(c.threads, static_cast<int>(settings.size()), [&](int k) {
    const std::uint64_t seed = val ? 5'300'000 + 100 * static_cast<std::uint64_t>(k) : 900000 + static_cast<std::uint64_t>(k);
    rc[sz(k)] = real_cover(setting_params(e, settings[sz(k)], seed), 150, 300);
  });
  std::vector<std::string> rows;
  for (const auto& m : models) {
    Runtime rt(m.m);
    std::vector<CoverRun> cr(settings.size());
    parallel(c.threads, static_cast<int>(settings.size()), [&](int k) {
      const std::vector<float> ctl(settings[sz(k)].begin(), settings[sz(k)].end());
      cr[sz(k)] = cover_run(runtime_clip(rt.e, ctl, -1, base + static_cast<std::uint64_t>(k), seconds * 30), rc[sz(k)]);
    });
    double all = 0, longest = 0;
    int hit = 0;
    for (std::size_t k = 0; k < settings.size(); ++k) {
      const auto& s = settings[k];
      rows.push_back(std::format("{},{},{},{},{:.2f},{:.2f},{:.2f},{:.3f},{:.3f},{:.1f},{:.1f}", ename(e), m.name, base, k, s[0], s[1], s[2], rc[k], cr[k].mean,
                                 100 * cr[k].below, cr[k].longest));
      all += cr[k].below / static_cast<double>(settings.size());
      longest = std::max(longest, cr[k].longest);
      hit += cr[k].below > 0 ? 1 : 0;
      std::println("{} {} setting {} ({:.2f}, {:.2f}, {:.2f}): real cover {:.3f}, cover {:.3f}, frames below a quarter {:.1f}%, longest {:.1f} s", ename(e), m.name, k,
                   s[0], s[1], s[2], rc[k], cr[k].mean, 100 * cr[k].below, cr[k].longest);
    }
    std::println("{} {} (seeds {} + k): {} of {} settings, {:.1f}% of all frames, longest {:.1f} s", ename(e), m.name, base, hit, settings.size(), 100 * all, longest);
  }
  merge_rows(c.results / std::format("l_collapse_{}{}.csv", val ? "val_" : "", base), "effect,model,seed_base,setting,intensity,second,turbulence,real_cover,cover,pct_below_quarter,longest_s",
             rows, ename(e));
}

// --- cost: ms per 128 x 128 frame, models interleaved -------------------------------------------------------------------

double thread_ms() {
  timespec ts{};
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
  return static_cast<double>(ts.tv_sec) * 1e3 + static_cast<double>(ts.tv_nsec) * 1e-6;
}
// Thread CPU time per frame through nvfx_render at controls (0.6, 0.5, 0.6), 200 frames after the first: per repetition
// the median over frames of each model, the models in turn; reported: the median over repetitions, its range, and each
// model's paired difference from the first (the current model).
void step_cost(const Ctx& c, sim::Effect e, const tools::Args& a) {
  (void)c;
  auto models = read_entries(a.need("current"), a.str("cand", ""));
  if (a.has("fire")) models.push_back({"fire", a.str("fire"), load_or_throw(a.str("fire")), file_sha256(a.str("fire"))});
  const int reps = a.i("reps", 7), size = a.i("size", 128), frames = 200;
  std::vector<std::unique_ptr<Runtime>> rts;
  for (const auto& m : models) rts.push_back(std::make_unique<Runtime>(m.m));
  std::vector<std::vector<double>> med(models.size());
  for (int r = 0; r < reps; ++r) {
    for (std::size_t k = 0; k < models.size(); ++k) {
      nvfx_instance* in = nullptr;
      if (nvfx_instance_create(rts[k]->e, size, &in) != NVFX_OK) throw std::runtime_error("instance");
      const float ctl[3] = {0.6f, 0.5f, 0.6f};
      nvfx_instance_set_controls(in, ctl, 3);
      nvfx_instance_set_seed(in, 77 + static_cast<std::uint64_t>(r));
      std::vector<std::uint8_t> buf(sz(size) * sz(size) * 4);
      nvfx_render(in, 0.0, buf.data(), sz(size) * 4);
      std::vector<double> ms;
      for (int f = 1; f <= frames; ++f) {
        const double t0 = thread_ms();
        nvfx_render(in, f / 30.0, buf.data(), sz(size) * 4);
        ms.push_back(thread_ms() - t0);
      }
      nvfx_instance_free(in);
      std::ranges::sort(ms);
      med[k].push_back(ms[ms.size() / 2]);
    }
  }
  const auto median = [](std::vector<double> v) {
    std::ranges::sort(v);
    return v[v.size() / 2];
  };
  std::vector<std::string> rows;
  for (std::size_t k = 0; k < models.size(); ++k) {
    std::vector<double> d;
    for (int r = 0; r < reps; ++r) d.push_back(med[k][sz(r)] - med[0][sz(r)]);
    const auto [lo, hi] = std::ranges::minmax(med[k]);
    std::println("cost {} {} at {} px: {:.3f} ms (range {:.3f} to {:.3f}), minus current {:+.3f} ms (median of {} paired repetitions)", ename(e), models[k].name, size,
                 median(med[k]), lo, hi, median(d), reps);
    rows.push_back(std::format("{},{},{},{:.4f},{:.4f},{:.4f},{:+.4f},{}", ename(e), models[k].name, size, median(med[k]), lo, hi, median(d), reps));
  }
  if (a.has("out")) merge_rows(a.str("out"), "effect,model,size,median_ms,min_ms,max_ms,minus_current_ms,reps", rows, ename(e));
}

// --- look: a sheet of frames played as shipped, a real run first ---------------------------------------------------------

void step_look(const Ctx& c, sim::Effect e, const tools::Args& a) {
  (void)c;
  const auto ctl = tools::parse_floats(a.need("setting"));
  if (ctl.size() != 3) throw std::invalid_argument("--setting i,w,t");
  const std::uint64_t seed = a.u64("seed", 123);
  const int frames = a.i("frames", 300), step = a.i("every", 30), scale = a.i("scale", 1);
  std::vector<int> cols;
  for (int f = step - 1; f < frames; f += step) cols.push_back(f);
  const auto pick = [&](const Clip& src) {
    Clip o;
    o.allocate(kSize, static_cast<int>(cols.size()));
    for (std::size_t q = 0; q < cols.size(); ++q) std::ranges::copy(src.frame(cols[q]), o.frame(static_cast<int>(q)).begin());
    return o;
  };
  std::vector<Clip> rows;
  {
    sim::Fluid f(setting_params(e, {ctl[0], ctl[1], ctl[2]}, seed));
    for (int i = 0; i < 150; ++i) f.step_frame();
    Clip real;
    real.allocate(kSize, frames);
    for (int i = 0; i < frames; ++i) {
      f.step_frame();
      f.render(real.frame(i));
    }
    rows.push_back(pick(real));
  }
  for (const auto& item : split_list(a.need("models"), ',')) {
    const auto eq = item.find('=');
    Runtime rt(load_or_throw(item.substr(eq + 1)));
    rows.push_back(pick(runtime_clip(rt.e, ctl, -1, seed + 1, frames)));
    std::println("look: {} {}", ename(e), item.substr(0, eq));
  }
  std::vector<const Clip*> rp;
  for (const auto& r : rows) rp.push_back(&r);
  if (auto w = write_png(a.need("out"), comparison_sheet(rp, static_cast<int>(cols.size()), Background::black, scale)); !w) throw std::runtime_error(w.error());
}
}  // namespace

int main(int argc, char** argv) {
  try {
    tools::Args a(argc, argv, {"no-neural", "no-csd", "only-neural", "empty-start"});
    if (a.positional().empty()) {
      std::println("nvfx_swirl diag|dstats|column --effect E [--root DIR] [--results DIR] [--threads N]");
      return 2;
    }
    Ctx c;
    const char* env = std::getenv("NEURALVFX_DATA");
    c.data = a.str("root", env ? env : "/root/nvfx-data");
    c.results = a.str("results", "results/experiments/swirl");
    c.threads = a.i("threads", 1);
    const std::string step = a.positional()[0];
    if (step == "graft" || step == "set") {
      if (step == "graft") step_graft(a);
      else step_set(a);
      return 0;
    }
    const sim::Effect e = effect_of(a.need("effect"));
    if (step == "diag") step_diag(c, e, a);
    else if (step == "dstats") step_dstats(c, e, a);
    else if (step == "coarse") step_coarse(c, e, a);
    else if (step == "train") step_train(c, e, a);
    else if (step == "advect") step_advect(c, e, a);
    else if (step == "column") step_column(c, e, a);
    else if (step == "column-figure") step_column_figure(c, e, a);
    else if (step == "val") step_val(c, e, a);
    else if (step == "test") step_test(c, e, a);
    else if (step == "collapse") step_collapse(c, e, a);
    else if (step == "cost") step_cost(c, e, a);
    else if (step == "look") step_look(c, e, a);
    else throw std::invalid_argument("unknown step " + step);
    a.warn_unused();
  } catch (const std::exception& e) {
    std::println(stderr, "nvfx_swirl: {}", e.what());
    return 1;
  }
  return 0;
}
