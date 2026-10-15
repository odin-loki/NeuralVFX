// nvfx_fix: study O (docs/REPORT.md §16): the most visible flaws of the shipped effects and demos, taken apart.
//
//   nvfx_fix bands --effect E --model F [--detail "swirl=0"] [--variants "p2:passes=2/c0:contrast=0"] [--runs 4]
//                  [--first 0] [--frames 90] [--rows 16,80] [--cols 56,72] [--out DIR] [--threads 1]
//       Horizontal banding in a falling sheet (the learned waterfall's, EFFECTS_WATER.md §1). On training runs (the
//       effect's salt-1 recipe runs first .. first + runs - 1: diagnosis only), the power spectrum along y of the frames'
//       luminance in a window (image rows [rows), columns [cols) of the 128-pixel frame; Hann window, each column's mean
//       taken off; mean over columns and frames) for each way of making the frames:
//         real               the simulator's own frames (after 100 frames of warm-up)
//         shipped_V          the runtime as an engine plays it (shards from the stored start points at the run's
//                            controls, a seed of its own; float), with variant V's detail layer
//         true_V             the reference rollout from the run's true state (coarse and fine fields, the run's seed),
//                            V's detail layer, the learned renderer
//         true_sim_V         the same states drawn by the simulator's renderer (the learned renderer taken out)
//         coarse_up          the reference rollout's coarse state alone (bilinear), the simulator's renderer (the
//                            learned body alone, no detail layer)
//         oracle_V           the run's true coarse state of every frame moving V's detail layer from the true fine
//                            fields, the simulator's renderer (the detail layer alone, with a perfect stepper)
//         oracle_up          the true coarse state alone (bilinear), the simulator's renderer
//       The first variant is "played": the model with --detail applied (as the library plays it). Each further variant
//       is the played model with its own settings on top. Writes DIR/bands_spectrum.csv (every method's spectrum, per
//       run), DIR/bands_summary.csv (per method: the excess vertical power over the real frames' in log10, averaged over
//       periods of 2.5 to 8 pixels, and the period of its largest excess), DIR/bands_frames.png (run 0, a window of the
//       frames, every method) and DIR/bands_spectrum.png (the spectra, run means).
//
//   nvfx_fix strip --model F --controls a,b,c [--detail ...] [--variants ...] [--size 256] [--times 2,4,6] [--seed S]
//                  [--crop x,y,w,h] [--scale 1] [--bg 36,40,50,13,14,17] [--precision float|int8] --out F.png
//       Frames of the runtime as an engine plays it, one row per variant (the first: the model with --detail), one column
//       per time; optionally a crop of each frame, scaled up (nearest). For the before and after figures.
//
// Seeds: study O's range, 3,000,000 to 3,499,999 (plays of `bands`: 3,000,000 + 100 r + k).
#include "args.hpp"
#include "label_font.hpp"

#include <neuralfx/clip.hpp>
#include <neuralfx/image_io.hpp>
#include <neuralfx/nvfx.h>
#include <neuralfx/rollout.hpp>
#include <neuralfx/rollout_train.hpp>
#include <neuralfx/sim.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <numbers>
#include <print>
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

std::vector<std::string> split_list(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  for (std::string x; std::getline(ss, x, sep);)
    if (!x.empty()) out.push_back(x);
  return out;
}

rollout::Model load_or_throw(const fs::path& p) {
  auto m = rollout::load_model(p);
  if (!m) throw std::runtime_error(std::format("{}: {}", p.string(), m.error()));
  return std::move(*m);
}

// "key=value;key=value" (or with commas) on a model's detail layer: the keys of a script's `effect` line and the
// detail layer's other constants.
void set_detail(rollout::Model& m, const std::string& spec) {
  std::string s = spec;
  std::ranges::replace(s, ',', ';');
  for (const auto& kv : split_list(s, ';')) {
    const auto eq = kv.find('=');
    if (eq == std::string::npos) throw std::invalid_argument("detail: key=value, not " + kv);
    const std::string k = kv.substr(0, eq);
    const float v = std::stof(kv.substr(eq + 1));
    rollout::DetailSpec& d = m.detail;
    if (k == "swirl") d.swirl = v;
    else if (k == "swirl_scale") d.swirl_scale = v;
    else if (k == "swirl_rate") d.swirl_rate = v;
    else if (k == "swirl_ramp") d.swirl_ramp = v;
    else if (k == "contrast") d.contrast = v;
    else if (k == "grow") d.grow = v;
    else if (k == "advect") d.advect = v;
    else if (k == "soften") d.soften = v;
    else if (k == "passes") d.passes = std::max(1, static_cast<int>(v));
    else throw std::invalid_argument("detail: unknown key " + k);
  }
}

struct Variant {
  std::string name;
  rollout::Model m;
};
// "name:key=v;key=v/name:..." on top of `base`
std::vector<Variant> variants_of(const rollout::Model& base, const std::string& played, const std::string& spec) {
  std::vector<Variant> v;
  v.push_back({played, base});
  for (const auto& item : split_list(spec, '/')) {
    const auto colon = item.find(':');
    Variant x{item.substr(0, colon), base};
    if (colon != std::string::npos) set_detail(x.m, item.substr(colon + 1));
    v.push_back(std::move(x));
  }
  return v;
}

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

// The runtime as an engine plays it: shards from the stored start points, a seed; frames [skip, skip + frames).
Clip runtime_clip(nvfx_effect* e, std::span<const float> controls, std::uint64_t seed, int skip, int frames, int size,
                  nvfx_precision prec) {
  nvfx_instance* in = nullptr;
  if (nvfx_instance_create(e, size, &in) != NVFX_OK) throw std::runtime_error("instance");
  nvfx_instance_set_precision(in, prec);
  nvfx_instance_set_controls(in, controls.data(), static_cast<int>(controls.size()));
  nvfx_instance_set_seed(in, seed);
  Clip c;
  c.allocate(size, frames);
  c.fps = 30.f;
  std::vector<std::uint8_t> scratch(sz(size) * sz(size) * 4);
  for (int f = 0; f < skip + frames; ++f) {
    std::uint8_t* out = f >= skip ? c.frame(f - skip).data() : scratch.data();
    nvfx_render(in, f / 30.0, out, sz(size) * 4);
  }
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

std::vector<float> upsample(const std::vector<float>& coarse, int channels, int c) {
  std::vector<float> o(sz(kSize) * sz(kSize));
  const float k = fl(kSize) / fl(kRes);
  for (int y = 0; y < kSize; ++y) {
    for (int x = 0; x < kSize; ++x) {
      const float xs = std::clamp((fl(x) + 0.5f) / k - 0.5f, 0.f, fl(kRes - 1)), ys = std::clamp((fl(y) + 0.5f) / k - 0.5f, 0.f, fl(kRes - 1));
      const int x0 = std::min(static_cast<int>(xs), kRes - 2), y0 = std::min(static_cast<int>(ys), kRes - 2);
      const float fx = xs - fl(x0), fy = ys - fl(y0);
      const auto at = [&](int xx, int yy) { return coarse[(sz(yy) * sz(kRes) + sz(xx)) * sz(channels) + sz(c)]; };
      o[sz(y) * sz(kSize) + sz(x)] = (1.f - fy) * ((1.f - fx) * at(x0, y0) + fx * at(x0 + 1, y0)) + fy * ((1.f - fx) * at(x0, y0 + 1) + fx * at(x0 + 1, y0 + 1));
    }
  }
  return o;
}

void to_bytes(const std::vector<float>& rgba, std::span<std::uint8_t> out) {
  for (std::size_t q = 0; q < rgba.size(); ++q) out[q] = static_cast<std::uint8_t>(std::clamp(rgba[q], 0.f, 1.f) * 255.f + 0.5f);
}

// --- the spectrum along y ---------------------------------------------------------------------------------------------------

// Mean power spectrum along y of the luminance (premultiplied, over black) in rows [r0, r1), columns [c0, c1): each
// column's profile has its mean taken off and a Hann window applied; |DFT|^2 at frequencies 1 .. N / 2 (cycles per N
// rows), mean over the columns and the frames.
std::vector<double> vspec(const Clip& cl, int r0, int r1, int c0, int c1) {
  const int N = r1 - r0, H = N / 2;
  std::vector<double> P(sz(H) + 1, 0.0), col(sz(N)), w(sz(N));
  for (int i = 0; i < N; ++i) w[sz(i)] = 0.5 - 0.5 * std::cos(2.0 * std::numbers::pi * (i + 0.5) / N);
  int n = 0;
  for (int f = 0; f < cl.frames; ++f) {
    const auto px = cl.frame(f);
    for (int x = c0; x < c1; ++x) {
      double mean = 0;
      for (int y = r0; y < r1; ++y) {
        const std::uint8_t* p = px.data() + (sz(y) * sz(cl.size) + sz(x)) * 4;
        col[sz(y - r0)] = 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
        mean += col[sz(y - r0)];
      }
      mean /= N;
      for (int k = 1; k <= H; ++k) {
        double re = 0, im = 0;
        for (int i = 0; i < N; ++i) {
          const double a = 2.0 * std::numbers::pi * k * i / N, v = (col[sz(i)] - mean) * w[sz(i)];
          re += v * std::cos(a);
          im -= v * std::sin(a);
        }
        P[sz(k)] += (re * re + im * im) / N;
      }
      ++n;
    }
  }
  for (double& p : P) p /= std::max(1, n);
  return P;
}

// How much the edges of the picture's coverage line up with the frame's axes: over every pixel and frame, the alpha
// channel's gradient g (central differences), sum |g|^2 cos(4 theta) / sum |g|^2, theta its angle. 0 for edges at every
// angle alike, 1 if every edge is horizontal or vertical, -1 if every one is diagonal (boxy shapes: towards 1).
double axis_index(const Clip& cl) {
  double num = 0, den = 0;
  const int S = cl.size;
  for (int f = 0; f < cl.frames; ++f) {
    const auto px = cl.frame(f);
    const auto a = [&](int x, int y) { return static_cast<double>(px[(sz(y) * sz(S) + sz(x)) * 4 + 3]) / 255.0; };
    for (int y = 1; y < S - 1; ++y)
      for (int x = 1; x < S - 1; ++x) {
        const double gx = 0.5 * (a(x + 1, y) - a(x - 1, y)), gy = 0.5 * (a(x, y + 1) - a(x, y - 1));
        const double g2 = gx * gx + gy * gy;
        if (g2 <= 0) continue;
        num += (gx * gx * gx * gx - 6.0 * gx * gx * gy * gy + gy * gy * gy * gy) / g2;
        den += g2;
      }
  }
  return den > 0 ? num / den : 0.0;
}

// --- a line chart (log10 power against the period) -------------------------------------------------------------------------

using Rgb = std::array<std::uint8_t, 3>;
void put(Image& img, int x, int y, Rgb c) {
  if (x < 0 || y < 0 || x >= img.width || y >= img.height) return;
  std::uint8_t* p = img.pixel(x, y);
  p[0] = c[0];
  p[1] = c[1];
  p[2] = c[2];
  p[3] = 255;
}
void line(Image& img, int x0, int y0, int x1, int y1, Rgb c) {
  const int n = std::max({std::abs(x1 - x0), std::abs(y1 - y0), 1});
  for (int i = 0; i <= n; ++i) {
    const int x = x0 + (x1 - x0) * i / n, y = y0 + (y1 - y0) * i / n;
    put(img, x, y, c);
    put(img, x, y + 1, c);
  }
}
void text(Image& img, std::string_view s, int x, int y, Rgb c) {
  tools::draw_text(s, x, y, 2, [&](int px, int py) { put(img, px, py, c); });
}

Image spectrum_chart(const std::vector<std::string>& names, const std::vector<std::vector<double>>& spectra, int N, const std::string& title) {
  constexpr int W = 980, Hh = 560, L = 90, R = 300, T = 50, B = 70;
  Image img;
  img.allocate(W, Hh);
  for (int y = 0; y < Hh; ++y)
    for (int x = 0; x < W; ++x) put(img, x, y, {250, 250, 250});
  double lo = 1e300, hi = -1e300;
  for (const auto& s : spectra)
    for (std::size_t k = 1; k < s.size(); ++k) {
      if (s[k] <= 0) continue;
      lo = std::min(lo, std::log10(s[k]));
      hi = std::max(hi, std::log10(s[k]));
    }
  lo = std::floor(lo);
  hi = std::ceil(hi);
  const int H = N / 2;
  const auto X = [&](int k) { return L + static_cast<int>(std::lround((std::log2(static_cast<double>(k)) / std::log2(static_cast<double>(H))) * (W - L - R))); };
  const auto Y = [&](double v) { return T + static_cast<int>(std::lround((hi - v) / (hi - lo) * (Hh - T - B))); };
  const Rgb axis{60, 60, 60}, grid{215, 215, 215};
  for (double v = lo; v <= hi + 1e-9; v += 1.0) {
    line(img, L, Y(v), W - R, Y(v), grid);
    text(img, std::format("{:.0f}", v), 20, Y(v) - 6, axis);
  }
  for (const double period : {32.0, 16.0, 8.0, 4.0, 2.0}) {
    const int k = static_cast<int>(std::lround(N / period));
    if (k < 1 || k > H) continue;
    line(img, X(k), T, X(k), Hh - B, grid);
    text(img, std::format("{:.0f}", period), X(k) - 8, Hh - B + 10, axis);
  }
  line(img, L, Hh - B, W - R, Hh - B, axis);
  line(img, L, T, L, Hh - B, axis);
  text(img, "PERIOD ALONG Y (PIXELS OF A 128-PIXEL FRAME)", L, Hh - 30, axis);
  text(img, "LOG10 POWER", 4, 18, axis);
  text(img, title, L + 140, 18, axis);
  static constexpr std::array<Rgb, 10> cols{{{20, 20, 20},    {214, 39, 40},  {31, 119, 180}, {44, 160, 44},  {255, 127, 14},
                                              {148, 103, 189}, {140, 86, 75},  {23, 190, 207}, {227, 119, 194}, {188, 189, 34}}};
  for (std::size_t m = 0; m < spectra.size(); ++m) {
    const Rgb c = cols[m % cols.size()];
    for (int k = 1; k < H; ++k) {
      if (spectra[m][sz(k)] <= 0 || spectra[m][sz(k + 1)] <= 0) continue;
      line(img, X(k), Y(std::log10(spectra[m][sz(k)])), X(k + 1), Y(std::log10(spectra[m][sz(k + 1)])), c);
    }
    const int ly = T + 10 + static_cast<int>(m) * 26;
    line(img, W - R + 20, ly + 6, W - R + 50, ly + 6, c);
    std::string up = names[m];
    std::ranges::transform(up, up.begin(), [](unsigned char ch) { return static_cast<char>(std::toupper(ch)); });
    text(img, up, W - R + 58, ly, c);
  }
  return img;
}

// --- bands -------------------------------------------------------------------------------------------------------------------

sim::Params run_params(sim::Effect e, std::uint64_t salt, std::uint64_t index) {
  rollout::SimRecipe r = rollout::recipe_for(e);
  r.salt = salt;
  return rollout::recipe_run(r, index);
}

void step_bands(const tools::Args& a) {
  sim::Effect e{};
  if (!sim::parse_effect(a.need("effect"), e)) throw std::invalid_argument("unknown effect");
  rollout::Model M = load_or_throw(a.need("model"));
  if (a.has("detail")) set_detail(M, a.str("detail"));
  const auto vars = variants_of(M, "played", a.str("variants", ""));
  const int n_runs = a.i("runs", 4), first = a.i("first", 0), K = a.i("frames", 90), warm = 100, threads = a.i("threads", 1);
  const int show = a.i("show-run", first);  // the run whose frames the figure shows
  const auto rows = split_list(a.str("rows", "16,80"), ','), cols = split_list(a.str("cols", "56,72"), ',');
  const int r0 = std::stoi(rows.at(0)), r1 = std::stoi(rows.at(1)), c0 = std::stoi(cols.at(0)), c1 = std::stoi(cols.at(1));
  const fs::path out = a.str("out", "bands");
  fs::create_directories(out);
  std::vector<std::string> methods = {"real"};
  for (const auto& v : vars) methods.push_back("shipped_" + v.name);
  for (const auto& v : vars) methods.push_back("true_" + v.name);
  for (const auto& v : vars) methods.push_back("true_sim_" + v.name);
  methods.push_back("coarse_up");
  for (const auto& v : vars) methods.push_back("oracle_" + v.name);
  methods.push_back("oracle_up");
  const std::size_t NM = methods.size();
  const auto idx = [&](const std::string& n) { return static_cast<std::size_t>(std::ranges::find(methods, n) - methods.begin()); };
  std::vector<std::vector<std::vector<double>>> spec(NM, std::vector<std::vector<double>>(sz(n_runs)));
  std::vector<std::vector<double>> axis(NM, std::vector<double>(sz(n_runs))), cover(NM, std::vector<double>(sz(n_runs)));
  std::vector<Clip> shown(NM);
  const auto t0 = std::chrono::steady_clock::now();
  parallel(threads, n_runs, [&](int r) {
    const sim::Params p = run_params(e, 1, static_cast<std::uint64_t>(first + r));
    const std::vector<float> ctl{p.intensity, p.wind, p.turbulence};
    sim::Fluid truth(p);
    for (int i = 0; i < warm; ++i) truth.step_frame();
    const sim::State st = truth.state();
    std::vector<Clip> clips(NM);
    for (auto& c : clips) {
      c.allocate(kSize, K);
      c.fps = 30.f;
    }
    std::vector<std::vector<float>> true_coarse(sz(K));
    for (int i = 0; i < K; ++i) {
      truth.step_frame();
      truth.render(clips[idx("real")].frame(i));
      true_coarse[sz(i)].resize(sz(kRes) * kRes * rollout::kPhys);
      rollout::coarse_from_sim(truth.state(), kRes, p.fps, true_coarse[sz(i)]);
      render_fields(p, upsample(true_coarse[sz(i)], rollout::kPhys, 2), upsample(true_coarse[sz(i)], rollout::kPhys, 3), clips[idx("oracle_up")].frame(i));
    }
    rollout::StartPoint sp;
    sp.controls = ctl;
    sp.seed = p.seed;
    sp.time = st.time;
    sp.coarse.resize(sz(kRes) * kRes * rollout::kPhys);
    rollout::coarse_from_sim(st, kRes, p.fps, sp.coarse);
    sp.fine_t = st.temp;
    sp.fine_d = st.soot;
    std::vector<float> rgba(sz(kSize) * kSize * 4);
    for (const auto& v : vars) {
      {  // as shipped: from 2 s on (the fine fields grown, the swirl ramped in)
        Runtime fx(v.m);
        clips[idx("shipped_" + v.name)] = runtime_clip(fx.e, ctl, 3'000'000 + 100 * static_cast<std::uint64_t>(r), 60, K, kSize, NVFX_PRECISION_FLOAT);
      }
      rollout::Model mt = v.m;
      mt.h.start_fine = kSize;
      mt.starts = {sp};
      rollout::State rs = rollout::start(mt, 0, kSize, ctl, sp.seed);
      const bool played = &v == &vars.front();
      for (int i = 0; i < K; ++i) {
        rollout::step(mt, rs, ctl, sp.seed);
        rollout::render(mt, rs, rgba);
        to_bytes(rgba, clips[idx("true_" + v.name)].frame(i));
        render_fields(p, rs.fine_t, rs.fine_d, clips[idx("true_sim_" + v.name)].frame(i));
        if (played) render_fields(p, upsample(rs.coarse, mt.h.channels(), 2), upsample(rs.coarse, mt.h.channels(), 3), clips[idx("coarse_up")].frame(i));
      }
      // the oracle stepper: the true coarse state of every frame moving this detail layer
      rollout::State os;
      os.res = kRes;
      os.size = kSize;
      os.time = st.time;
      os.since_start = 0.f;
      const int C = v.m.h.channels();
      os.coarse.assign(sz(kRes) * kRes * sz(C), 0.f);
      os.flow.assign(sz(kRes) * kRes * 2, 0.f);
      os.pressure.assign(sz(kRes) * kRes, 0.f);
      os.fine_t = st.temp;
      os.fine_d = st.soot;
      for (int i = 0; i < K; ++i) {
        for (int j = 0; j < kRes * kRes; ++j) {
          for (int q = 0; q < rollout::kPhys; ++q) os.coarse[sz(j) * sz(C) + sz(q)] = true_coarse[sz(i)][sz(j) * rollout::kPhys + sz(q)];
          os.flow[sz(j) * 2] = true_coarse[sz(i)][sz(j) * rollout::kPhys];
          os.flow[sz(j) * 2 + 1] = true_coarse[sz(i)][sz(j) * rollout::kPhys + 1];
        }
        rollout::detail_step(v.m, os, p.seed, ctl);
        os.time += 1.f / v.m.fps;
        os.since_start += 1.f / v.m.fps;
        render_fields(p, os.fine_t, os.fine_d, clips[idx("oracle_" + v.name)].frame(i));
      }
    }
    for (std::size_t m = 0; m < NM; ++m) {
      spec[m][sz(r)] = vspec(clips[m], r0, r1, c0, c1);
      axis[m][sz(r)] = axis_index(clips[m]);
      double al = 0;
      for (int f = 0; f < clips[m].frames; ++f)
        for (std::size_t q = 3; q < clips[m].frame(f).size(); q += 4) al += clips[m].frame(f)[q] / 255.0;
      cover[m][sz(r)] = al / (static_cast<double>(clips[m].frames) * kSize * kSize);
    }
    if (first + r == show) shown = std::move(clips);
    std::println("bands: run {} ({:.2f}, {:.2f}, {:.2f}) done ({:.0f} s)", first + r, ctl[0], ctl[1], ctl[2],
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    std::fflush(stdout);
  });
  const int N = r1 - r0, H = N / 2;
  std::ofstream cs(out / "bands_spectrum.csv");
  cs << "method,run,k,period,power\n";
  for (std::size_t m = 0; m < NM; ++m)
    for (int r = 0; r < n_runs; ++r)
      for (int k = 1; k <= H; ++k) cs << std::format("{},{},{},{:.3f},{:.6g}\n", methods[m], first + r, k, static_cast<double>(N) / k, spec[m][sz(r)][sz(k)]);
  // run means, and the excess over the real frames
  std::vector<std::vector<double>> mean(NM, std::vector<double>(sz(H) + 1, 0.0));
  for (std::size_t m = 0; m < NM; ++m)
    for (int r = 0; r < n_runs; ++r)
      for (int k = 1; k <= H; ++k) mean[m][sz(k)] += spec[m][sz(r)][sz(k)] / n_runs;
  {  // the edges' alignment with the frame's axes, per method: mean over runs, and each run
    std::ofstream ax(out / "axis_index.csv");
    ax << "method,run,axis_index\n";
    std::println("\nthe coverage's edges along the frame's axes (axis index: 0 at every angle alike, 1 all horizontal or vertical), mean over runs and per run:");
    for (std::size_t m = 0; m < NM; ++m) {
      double mu = 0;
      std::string line;
      for (int r = 0; r < n_runs; ++r) {
        mu += axis[m][sz(r)] / n_runs;
        line += std::format(" {:+.3f}", axis[m][sz(r)]);
        ax << std::format("{},{},{:.5f}\n", methods[m], first + r, axis[m][sz(r)]);
      }
      double cv = 0;
      for (int r = 0; r < n_runs; ++r) cv += cover[m][sz(r)] / n_runs;
      std::println("  {:<22} {:+.3f}:{}   (mean coverage {:.3f})", methods[m], mu, line, cv);
    }
  }
  std::ofstream su(out / "bands_summary.csv");
  su << "method,excess_log10_2p5_8px,excess_sd_runs,peak_period,peak_excess_log10\n";
  std::println("\n{} (runs {} to {}; rows {}..{}, columns {}..{}): vertical power over the real frames' (log10), periods 2.5 to 8 px",
               sim::effect_name(e), first, first + n_runs - 1, r0, r1, c0, c1);
  for (std::size_t m = 0; m < NM; ++m) {
    std::vector<double> per_run;
    for (int r = 0; r < n_runs; ++r) {
      double s = 0;
      int n = 0;
      for (int k = 1; k <= H; ++k) {
        const double period = static_cast<double>(N) / k;
        if (period < 2.5 || period > 8.0) continue;
        s += std::log10(std::max(1e-12, spec[m][sz(r)][sz(k)]) / std::max(1e-12, spec[0][sz(r)][sz(k)]));
        ++n;
      }
      per_run.push_back(s / std::max(1, n));
    }
    double mu = 0, var = 0;
    for (const double v : per_run) mu += v / n_runs;
    for (const double v : per_run) var += (v - mu) * (v - mu) / std::max(1, n_runs - 1);
    int kp = 1;
    double best = -1e300;
    for (int k = 2; k <= H; ++k) {
      const double ex = std::log10(std::max(1e-12, mean[m][sz(k)]) / std::max(1e-12, mean[0][sz(k)]));
      if (ex > best) best = ex, kp = k;
    }
    su << std::format("{},{:.3f},{:.3f},{:.2f},{:.3f}\n", methods[m], mu, std::sqrt(var), static_cast<double>(N) / kp, best);
    std::println("  {:<22} {:+.3f} (sd over runs {:.3f}); largest excess {:+.3f} at a period of {:.2f} px", methods[m], mu, std::sqrt(var), best,
                 static_cast<double>(N) / kp);
  }
  // Per run: the period of the largest excess over the real frames between 2 and 8 px (the bands' spacing), against how
  // far the falling water moves in a frame (the simulator's terminal speed for the waterfall and the rain shaft).
  std::ofstream pr(out / "bands_peaks.csv");
  pr << "method,run,intensity,travel_px,peak_period,peak_excess_log10\n";
  std::println("\nper run: the period of the largest excess between 2 and 8 px (and the excess), against the travel per frame");
  for (int r = 0; r < n_runs; ++r) {
    const sim::Params p = run_params(e, 1, static_cast<std::uint64_t>(first + r));
    const float fall = e == sim::Effect::rainshaft ? 80.f + 30.f * p.intensity : 85.f + 30.f * p.intensity;
    const double travel = fall / 30.0;
    std::string line = std::format("  run {:2d} I {:.2f} travel {:.2f} px:", first + r, p.intensity, travel);
    for (std::size_t m = 1; m < NM; ++m) {
      int kp = 0;
      double best = -1e300;
      for (int k = 1; k <= H; ++k) {
        const double period = static_cast<double>(N) / k;
        if (period < 2.0 || period > 8.0) continue;
        const double ex = std::log10(std::max(1e-12, spec[m][sz(r)][sz(k)]) / std::max(1e-12, spec[0][sz(r)][sz(k)]));
        if (ex > best) best = ex, kp = k;
      }
      pr << std::format("{},{},{:.3f},{:.3f},{:.3f},{:.3f}\n", methods[m], first + r, p.intensity, travel, static_cast<double>(N) / kp, best);
      if (methods[m].starts_with("shipped_") || methods[m] == "oracle_played") line += std::format("  {} {:.2f} ({:+.2f})", methods[m], static_cast<double>(N) / kp, best);
    }
    std::println("{}", line);
  }
  // figures: run 0's window, every method, at 4 frames; and the spectra
  {
    // columns: the methods of --figure-methods ("method=LABEL,...", default all), rows: frames across the run
    std::vector<std::pair<std::size_t, std::string>> fm;
    for (const auto& item : split_list(a.str("figure-methods", ""), ',')) {
      const auto eq = item.find('=');
      const std::string name = item.substr(0, eq);
      if (idx(name) >= NM) throw std::invalid_argument("--figure-methods: no method " + name);
      fm.push_back({idx(name), eq == std::string::npos ? name : item.substr(eq + 1)});
    }
    if (fm.empty())
      for (std::size_t m = 0; m < NM; ++m) fm.push_back({m, methods[m]});
    const int fw = c1 - c0 + 24, fh = r1 - r0 + 8, sc = 3, rows_n = a.i("figure-rows", 3), top = 16;
    Image img;
    img.allocate(static_cast<int>(fm.size()) * (fw * sc + 4) - 4, top + rows_n * (fh * sc + 4) - 4);
    for (auto& b : img.rgba) b = 255;
    std::vector<std::uint8_t> tile(sz(kSize) * kSize * 4);
    for (std::size_t c = 0; c < fm.size(); ++c) {
      std::string up = fm[c].second;
      std::ranges::transform(up, up.begin(), [](unsigned char ch) { return static_cast<char>(std::toupper(ch)); });
      tools::draw_text(up, static_cast<int>(c) * (fw * sc + 4) + 2, 4, 1, [&](int px, int py) {
        if (px < img.width && py < img.height) {
          std::uint8_t* q = img.pixel(px, py);
          q[0] = q[1] = q[2] = 30;
          q[3] = 255;
        }
      });
      for (int j = 0; j < rows_n; ++j) {
        const int f = (j + 1) * (K - 1) / rows_n;
        composite(shown[fm[c].first].frame(f), kSize, Background::black, tile);
        for (int y = 0; y < fh * sc; ++y)
          for (int x = 0; x < fw * sc; ++x) {
            const int sx = c0 - 12 + x / sc, sy = r0 - 4 + y / sc;
            if (sx < 0 || sy < 0 || sx >= kSize || sy >= kSize) continue;
            std::uint8_t* q = img.pixel(static_cast<int>(c) * (fw * sc + 4) + x, top + j * (fh * sc + 4) + y);
            std::copy_n(tile.data() + (sz(sy) * kSize + sz(sx)) * 4, 4, q);
          }
      }
    }
    if (auto w = write_png(out / "bands_frames.png", img); !w) std::println("figure: {}", w.error());
    std::vector<std::string> shown_names;
    std::vector<std::vector<double>> shown_spec;
    for (const std::string& want : split_list(a.str("chart", "real,shipped_played,true_sim_played,coarse_up,oracle_played"), ',')) {
      if (idx(want) >= NM) throw std::invalid_argument("--chart: no method " + want);
      shown_names.push_back(want);
      shown_spec.push_back(mean[idx(want)]);
    }
    if (auto w = write_png(out / "bands_spectrum.png", spectrum_chart(shown_names, shown_spec, N, std::string(sim::effect_name(e)) + ": THE SHEET'S ROWS")); !w)
      std::println("figure: {}", w.error());
  }
}

// --- peaks: the bands as an engine plays the effect, at any size ---------------------------------------------------------------

// How far the largest narrow peak of a spectrum stands above its neighbours, in log10 (0 for a smooth spectrum), and its
// bin: over bins k whose period N / k lies in [lo, hi], log10 P(k) less the mean of log10 P at k -+ 2 and k -+ 3.
std::pair<double, int> prominence(const std::vector<double>& P, int N, double lo, double hi) {
  const int H = static_cast<int>(P.size()) - 1;
  double best = -1e300;
  int kb = 0;
  for (int k = 4; k <= H - 3; ++k) {
    const double period = static_cast<double>(N) / k;
    if (period < lo || period > hi) continue;
    const auto lg = [&](int j) { return std::log10(std::max(1e-12, P[sz(j)])); };
    const double v = lg(k) - 0.25 * (lg(k - 3) + lg(k - 2) + lg(k + 2) + lg(k + 3));
    if (v > best) best = v, kb = k;
  }
  return {best, kb};
}

// The runtime as shipped at the controls of the effect's training runs (salt 1), each variant, at --size: the spectrum
// along y in the window (rows and columns given at 128 px, scaled to the size) and its largest narrow peak.
void step_peaks(const tools::Args& a) {
  sim::Effect e{};
  if (!sim::parse_effect(a.need("effect"), e)) throw std::invalid_argument("unknown effect");
  rollout::Model M = load_or_throw(a.need("model"));
  if (a.has("detail")) set_detail(M, a.str("detail"));
  const auto vars = variants_of(M, "played", a.str("variants", ""));
  const int n_runs = a.i("runs", 8), first = a.i("first", 0), K = a.i("frames", 90), size = a.i("size", 128), threads = a.i("threads", 1);
  const int f = size / kSize;
  const auto rows = split_list(a.str("rows", "16,80"), ','), cols = split_list(a.str("cols", "56,72"), ',');
  const int r0 = f * std::stoi(rows.at(0)), r1 = f * std::stoi(rows.at(1)), c0 = f * std::stoi(cols.at(0)), c1 = f * std::stoi(cols.at(1));
  const int N = r1 - r0;
  const fs::path out = a.str("out", "peaks");
  fs::create_directories(out);
  std::vector<std::vector<std::vector<double>>> spec(vars.size(), std::vector<std::vector<double>>(sz(n_runs)));
  parallel(threads, n_runs * static_cast<int>(vars.size()), [&](int j) {
    const int r = j / static_cast<int>(vars.size()), vi = j % static_cast<int>(vars.size());
    const sim::Params p = run_params(e, 1, static_cast<std::uint64_t>(first + r));
    const std::vector<float> ctl{p.intensity, p.wind, p.turbulence};
    Runtime fx(vars[sz(vi)].m);
    spec[sz(vi)][sz(r)] = vspec(runtime_clip(fx.e, ctl, 3'010'000 + 100 * static_cast<std::uint64_t>(r), 60, K, size, NVFX_PRECISION_FLOAT), r0, r1, c0, c1);
  });
  std::ofstream cs(out / std::format("peaks_{}.csv", size));
  cs << "variant,run,intensity,prominence_log10,period_px\n";
  std::println("{} at {} px (runs {} to {}): the largest narrow peak of the spectrum along y (log10 above its neighbours) and its period",
               sim::effect_name(e), size, first, first + n_runs - 1);
  for (std::size_t vi = 0; vi < vars.size(); ++vi) {
    double mean = 0;
    std::string line;
    for (int r = 0; r < n_runs; ++r) {
      const auto [v, k] = prominence(spec[vi][sz(r)], N, 2.0, 12.0 * f);
      const sim::Params p = run_params(e, 1, static_cast<std::uint64_t>(first + r));
      cs << std::format("{},{},{:.3f},{:.3f},{:.2f}\n", vars[vi].name, first + r, p.intensity, v, k > 0 ? static_cast<double>(N) / k : 0.0);
      mean += v / n_runs;
      line += std::format(" {:+.2f}@{:.1f}", v, k > 0 ? static_cast<double>(N) / k : 0.0);
    }
    std::println("  {:<10} mean {:+.3f}:{}", vars[vi].name, mean, line);
  }
}

// --- cover: how much of the frame an effect covers over time ----------------------------------------------------------------

// At the controls of training runs (salt 1), the mean alpha of each second: the simulator's run (from an empty grid, its
// own seed), the runtime as shipped (shards from the stored start points; a seed of study O's) and the runtime as one
// rollout from the nearest start point (no shards). Development only.
void step_cover(const tools::Args& a) {
  sim::Effect e{};
  if (!sim::parse_effect(a.need("effect"), e)) throw std::invalid_argument("unknown effect");
  rollout::Model M = load_or_throw(a.need("model"));
  if (a.has("detail")) set_detail(M, a.str("detail"));
  const int n_runs = a.i("runs", 6), first = a.i("first", 0), secs = a.i("seconds", 20), threads = a.i("threads", 1);
  std::vector<std::array<std::vector<double>, 3>> res(sz(n_runs));
  Runtime fx(M);
  parallel(threads, n_runs, [&](int r) {
    const sim::Params p = run_params(e, 1, static_cast<std::uint64_t>(first + r));
    const std::vector<float> ctl{p.intensity, p.wind, p.turbulence};
    std::vector<std::uint8_t> px(sz(kSize) * kSize * 4);
    const auto mean_alpha = [&] {
      double s = 0;
      for (std::size_t q = 3; q < px.size(); q += 4) s += px[q] / 255.0;
      return s / (kSize * kSize);
    };
    sim::Fluid f(p);
    for (int t = 0; t < secs; ++t) {
      double acc = 0;
      for (int i = 0; i < 30; ++i) {
        f.step_frame();
        f.render(px);
        acc += mean_alpha() / 30;
      }
      res[sz(r)][0].push_back(acc);
    }
    for (int mode = 1; mode <= 2; ++mode) {
      nvfx_instance* in = nullptr;
      if (nvfx_instance_create(fx.e, kSize, &in) != NVFX_OK) throw std::runtime_error("instance");
      nvfx_instance_set_precision(in, NVFX_PRECISION_FLOAT);
      nvfx_instance_set_controls(in, ctl.data(), 3);
      nvfx_instance_set_seed(in, 3'020'000 + static_cast<std::uint64_t>(r));
      if (mode == 2) nvfx_instance_set_drift(in, 0.f);
      for (int t = 0; t < secs; ++t) {
        double acc = 0;
        for (int i = 0; i < 30; ++i) {
          nvfx_render(in, (t * 30 + i) / 30.0, px.data(), sz(kSize) * 4);
          acc += mean_alpha() / 30;
        }
        res[sz(r)][sz(mode)].push_back(acc);
      }
      nvfx_instance_free(in);
    }
  });
  const char* names[3] = {"simulator", "shards", "one rollout"};
  for (int m = 0; m < 3; ++m) {
    std::string line;
    for (int t = 0; t < secs; ++t) {
      double v = 0;
      for (int r = 0; r < n_runs; ++r) v += res[sz(r)][sz(m)][sz(t)] / n_runs;
      line += std::format(" {:.3f}", v);
    }
    std::println("{:<12} mean alpha by second:{}", names[m], line);
  }
}

// --- strip -------------------------------------------------------------------------------------------------------------------

void step_strip(const tools::Args& a) {
  rollout::Model M = load_or_throw(a.need("model"));
  if (a.has("detail")) set_detail(M, a.str("detail"));
  const auto vars = variants_of(M, "played", a.str("variants", ""));
  std::vector<float> ctl;
  for (const auto& s : split_list(a.need("controls"), ',')) ctl.push_back(std::stof(s));
  std::vector<double> times;
  for (const auto& s : split_list(a.str("times", "2,4,6"), ',')) times.push_back(std::stod(s));
  const int size = a.i("size", 256), scale = a.i("scale", 1);
  const std::uint64_t seed = a.u64("seed", 3'100'000);
  const nvfx_precision prec = a.str("precision", "float") == "int8" ? NVFX_PRECISION_INT8 : NVFX_PRECISION_FLOAT;
  int cx = 0, cy = 0, cw = size, ch = size;
  if (a.has("crop")) {
    const auto v = split_list(a.str("crop"), ',');
    cx = std::stoi(v.at(0)), cy = std::stoi(v.at(1)), cw = std::stoi(v.at(2)), ch = std::stoi(v.at(3));
  }
  std::array<float, 6> bg{36, 40, 50, 13, 14, 17};
  if (a.has("bg")) {
    const auto v = split_list(a.str("bg"), ',');
    for (std::size_t i = 0; i < 6; ++i) bg[i] = std::stof(v.at(i));
  }
  const int W = cw * scale, Hc = ch * scale, gap = 4;
  Image img;
  img.allocate(static_cast<int>(times.size()) * (W + gap) - gap, static_cast<int>(vars.size()) * (Hc + gap) - gap);
  for (auto& b : img.rgba) b = 255;
  const int last = static_cast<int>(std::lround(times.back() * 30.0));
  for (std::size_t vi = 0; vi < vars.size(); ++vi) {
    Runtime fx(vars[vi].m);
    nvfx_instance* in = nullptr;
    if (nvfx_instance_create(fx.e, size, &in) != NVFX_OK) throw std::runtime_error("instance");
    nvfx_instance_set_precision(in, prec);
    nvfx_instance_set_controls(in, ctl.data(), static_cast<int>(ctl.size()));
    nvfx_instance_set_seed(in, seed);
    std::vector<std::uint8_t> px(sz(size) * sz(size) * 4);
    for (int f = 0; f <= last; ++f) {
      nvfx_render(in, f / 30.0, px.data(), sz(size) * 4);
      const auto it = std::ranges::find_if(times, [&](double t) { return std::lround(t * 30.0) == f; });
      if (it == times.end()) continue;
      const int col = static_cast<int>(it - times.begin());
      for (int y = 0; y < Hc; ++y) {
        const int sy = cy + y / scale;
        const float u = static_cast<float>(sy) / static_cast<float>(std::max(1, size - 1));
        for (int x = 0; x < W; ++x) {
          const int sx = cx + x / scale;
          if (sx < 0 || sy < 0 || sx >= size || sy >= size) continue;
          const std::uint8_t* p = px.data() + (sz(sy) * sz(size) + sz(sx)) * 4;
          std::uint8_t* q = img.pixel(col * (W + gap) + x, static_cast<int>(vi) * (Hc + gap) + y);
          const float al = static_cast<float>(p[3]) / 255.f;
          for (int k = 0; k < 3; ++k) {
            const float b = bg[sz(k)] + (bg[sz(k) + 3] - bg[sz(k)]) * u;
            q[k] = static_cast<std::uint8_t>(std::clamp(static_cast<float>(p[k]) + b * (1.f - al), 0.f, 255.f) + 0.5f);
          }
          q[3] = 255;
        }
      }
    }
    nvfx_instance_free(in);
  }
  if (auto w = write_png(a.need("out"), img); !w) throw std::runtime_error(w.error());
  std::println("strip: {} variants x {} times -> {}", vars.size(), times.size(), a.need("out"));
}

}  // namespace

int main(int argc, char** argv) {
  try {
    tools::Args a(argc, argv, {});
    if (a.positional().empty()) {
      std::println(stderr, "usage: nvfx_fix bands|peaks|strip ... (see the header of tools/nvfx_fix.cpp)");
      return 2;
    }
    const std::string step = a.positional()[0];
    if (step == "bands") step_bands(a);
    else if (step == "strip") step_strip(a);
    else if (step == "peaks") step_peaks(a);
    else if (step == "cover") step_cover(a);
    else throw std::invalid_argument("unknown step " + step);
    a.warn_unused();
  } catch (const std::exception& ex) {
    std::println(stderr, "nvfx_fix: {}", ex.what());
    return 1;
  }
  return 0;
}
