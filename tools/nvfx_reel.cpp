// nvfx_reel: demo videos of the learned effects (videos/README.md). Panels side by side, each an effect played through
// nvfx_render with its controls keyed over time, or the simulator's held-out run and the learned effect started from
// that run's true state; composited over a dark background, labelled with the 5 x 7 font of the gallery, and piped to
// ffmpeg (H.264, yuv420p, +faststart, no audio).
//
//   nvfx_reel --out F.mp4 --seconds S --panel SPEC [--panel SPEC ...] [--size 256] [--zoom 1.5] [--filter bilinear|nearest]
//             [--width 1280] [--height 720] [--columns N] [--title TEXT] [--note TEXT ...] [--caption TEXT ...] [--timer]
//             [--bars | --compact] [--seed 1] [--seed-base N] [--restart S] [--crf 26] [--bg R,G,B,R,G,B]
//
//   SPEC is LABEL|SOURCE[|KEYS[|DETAIL[|BG]]]:
//     FILE.nvfx                  a learned effect; KEYS "T:C1,C2,C3;T:C1,C2,C3;..." its controls at times T (seconds),
//                                linear in between, held before the first and after the last (one key: constant)
//     real:EFFECT:RUN            the simulator's held-out run RUN of EFFECT (study D's held-out runs, salt 2: their own
//                                settings and seeds), from its frame 100 (one-shot effects: frame 1)
//     learned:EFFECT:RUN:FILE    the learned effect FILE started from that run's true state at that frame (its coarse
//                                state and fine fields as the only start point, the run's noise seed), one continuous
//                                rollout: frame i of both comes i frames after the same state (study D's tracking, §6.4)
//   DETAIL "swirl=1,swirl_scale=9": the learned model's detail layer as a scene script's `effect` line sets it (swirl,
//          swirl_scale, swirl_rate, swirl_ramp, contrast, grow, passes), for FILE and learned panels (KEYS may be empty)
//   BG     "R,G,B,R,G,B": the panel's background gradient, top and bottom colours (default --bg's: a dark night sky,
//          36,40,50,13,14,17); a dark effect (ink) needs a light one
//   --seed-base N: real and learned panels take seed N + RUN for the held-out run's noise (its settings stay salt 2's),
//                as nvfx_experiment d-eval --seed-base does (the earth effects' test, docs/EFFECTS_EARTH.md: 6000000)
//   --restart S: real and learned panels start again every S seconds from the next run (RUN, RUN + 1, ...), one-shot
//                effects with the next seed (default: one-shots only, real and learned panels after their length and
//                0.5 s, effects after their length and 1 s).
//
// Learned effects play at the runtime's default precision (int8 for rollout effects) and drift (6 s shards); real and
// learned panels are 128 px (the simulator's size), others --size px. Each panel is composited over its background at
// its own size and scaled by --zoom (bilinear or nearest). Under each panel its controls are printed (--bars: drawn as
// bars, wind from the middle; --compact: on one line, by their initials). --timer prints the time in the title bar.
// One thread renders; ffmpeg gets two.
#include "label_font.hpp"

#include <neuralfx/nvfx.h>
#include <neuralfx/rollout.hpp>
#include <neuralfx/rollout_train.hpp>
#include <neuralfx/sim.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <format>
#include <memory>
#include <print>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
using namespace nfx;
using Rgb = std::array<std::uint8_t, 3>;
constexpr int kSimSize = 128;
constexpr float kFps = 30.f;

std::size_t sz(int v) { return static_cast<std::size_t>(v); }

std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  for (std::string x; std::getline(ss, x, sep);) out.push_back(x);
  return out;
}

struct Canvas {
  int w = 0, h = 0;
  std::vector<std::uint8_t> rgb;
  void fill(int x0, int y0, int ww, int hh, Rgb c) {
    for (int y = std::max(0, y0); y < std::min(h, y0 + hh); ++y)
      for (int x = std::max(0, x0); x < std::min(w, x0 + ww); ++x)
        for (int k = 0; k < 3; ++k) rgb[(sz(y) * sz(w) + sz(x)) * 3 + sz(k)] = c[sz(k)];
  }
  void text(std::string_view s, int x, int y, int scale, Rgb c) {
    tools::draw_text(s, x, y, scale, [&](int px, int py) {
      if (px >= 0 && py >= 0 && px < w && py < h)
        for (int k = 0; k < 3; ++k) rgb[(sz(py) * sz(w) + sz(px)) * 3 + sz(k)] = c[sz(k)];
    });
  }
};

// A panel's background: a vertical gradient from `top` to `bottom` (by default a dark night sky, so that both glowing
// and dark material show).
using Gradient = std::array<float, 6>;
constexpr Gradient kNight{36, 40, 50, 13, 14, 17};
Rgb background(int y, int h, const Gradient& g) {
  const float u = static_cast<float>(y) / static_cast<float>(std::max(1, h - 1));
  const auto mix = [u](float a, float b) { return static_cast<std::uint8_t>(std::lround(a + (b - a) * u)); };
  return {mix(g[0], g[3]), mix(g[1], g[4]), mix(g[2], g[5])};
}

Gradient parse_gradient(const std::string& text) {
  Gradient g{};
  const auto v = split(text, ',');
  if (v.size() != 6) throw std::invalid_argument("a background is R,G,B,R,G,B (top, bottom): " + text);
  for (std::size_t i = 0; i < 6; ++i) g[i] = std::stof(v[i]);
  return g;
}

struct Key {
  float t = 0;
  std::vector<float> c;
};

struct Panel {
  std::string label, kind;  // kind: "effect", "real" or "learned"
  std::vector<Key> keys;
  std::vector<float> controls;
  std::vector<std::string> names;
  int size = 0;
  // effect
  nvfx_effect* fx = nullptr;
  nvfx_instance* in = nullptr;
  std::uint64_t seed = 1;
  bool loops = true;
  float length = 0.f;  // a one-shot effect's seconds
  int cycle = 0;
  std::vector<std::pair<std::string, float>> detail;  // DETAIL
  Gradient bg = kNight;                                // BG
  // real and learned
  sim::Effect effect = sim::Effect::fire;
  int run0 = 0, run = -1;
  fs::path model;
  std::unique_ptr<sim::Fluid> truth;
  std::vector<std::uint8_t> rgba;

  ~Panel() {
    if (in) nvfx_instance_free(in);
    if (fx) nvfx_effect_free(fx);
  }
};

void load_names(Panel& p) {
  nvfx_effect_info info{};
  nvfx_effect_get_info(p.fx, &info);
  p.names.clear();
  for (int i = 0; i < info.n_controls; ++i) {
    const char* name = nvfx_effect_control_name(p.fx, i);
    p.names.emplace_back(name ? name : std::format("control {}", i));
  }
}

// A scene script's detail settings on a learned model (script_run.cpp reads the same keys from an `effect` line).
void set_detail(rollout::Model& m, const std::vector<std::pair<std::string, float>>& detail) {
  for (const auto& [k, v] : detail) {
    rollout::DetailSpec& d = m.detail;
    if (k == "swirl") d.swirl = v;
    else if (k == "swirl_scale") d.swirl_scale = v;
    else if (k == "swirl_rate") d.swirl_rate = v;
    else if (k == "swirl_ramp") d.swirl_ramp = v;
    else if (k == "contrast") d.contrast = v;
    else if (k == "grow") d.grow = v;
    else if (k == "passes") d.passes = std::max(1, static_cast<int>(v));
    else throw std::invalid_argument("no detail setting " + k);
  }
}

nvfx_effect* load_effect(const rollout::Model& m) {
  std::ostringstream os;
  if (auto w = rollout::save_model(os, m); !w) throw std::runtime_error(w.error());
  const std::string bytes = os.str();
  nvfx_effect* fx = nullptr;
  if (nvfx_effect_load_memory(bytes.data(), bytes.size(), &fx) != NVFX_OK) throw std::runtime_error("cannot load a changed model");
  return fx;
}

// Real and learned panels: (re)start at run `run` of the held-out set (its seed seed_base + run when seed_base > 0).
std::uint64_t g_seed_base = 0;
void start_run(Panel& p, int run) {
  p.run = run;
  rollout::SimRecipe r;
  r.effect = p.effect;
  r.salt = 2;  // study D's held-out runs
  sim::Params prm = rollout::recipe_run(r, static_cast<std::uint64_t>(run));
  if (g_seed_base > 0) prm.seed = g_seed_base + static_cast<std::uint64_t>(run);
  p.controls = {prm.intensity, prm.wind, prm.turbulence};
  p.names.clear();
  for (const std::string_view name : sim::control_names(p.effect)) p.names.emplace_back(name);
  auto fluid = std::make_unique<sim::Fluid>(prm);
  const int warm = sim::effect_loops(p.effect) ? 100 : 1;
  for (int i = 0; i < warm; ++i) fluid->step_frame();
  if (p.kind == "real") {
    p.truth = std::move(fluid);
    return;
  }
  const sim::State st = fluid->state();
  auto loaded = rollout::load_model(p.model);
  if (!loaded) throw std::runtime_error(loaded.error());
  rollout::Model m = std::move(*loaded);
  m.h.start_fine = kSimSize;
  rollout::StartPoint sp;
  sp.controls = p.controls;
  sp.seed = prm.seed;
  sp.time = st.time;
  sp.coarse.resize(sz(m.h.res) * sz(m.h.res) * rollout::kPhys);
  rollout::coarse_from_sim(st, m.h.res, prm.fps, sp.coarse);
  sp.fine_t = st.temp;
  sp.fine_d = st.soot;
  m.starts = {sp};
  set_detail(m, p.detail);
  if (p.in) nvfx_instance_free(p.in);
  if (p.fx) nvfx_effect_free(p.fx);
  p.in = nullptr;
  p.fx = nullptr;
  p.fx = load_effect(m);
  if (nvfx_instance_create(p.fx, kSimSize, &p.in) != NVFX_OK) throw std::runtime_error("cannot create an instance");
  nvfx_instance_set_controls(p.in, p.controls.data(), static_cast<int>(p.controls.size()));
  nvfx_instance_set_drift(p.in, 0.f);     // one continuous rollout
  nvfx_instance_set_variation(p.in, 0);   // from the true start point, with the run's seed
}

std::unique_ptr<Panel> make_panel(const std::string& spec, int size, std::uint64_t seed, const Gradient& bg) {
  const auto f = split(spec, '|');
  if (f.size() < 2) throw std::invalid_argument("--panel LABEL|SOURCE[|KEYS[|DETAIL[|BG]]]: " + spec);
  auto pp = std::make_unique<Panel>();
  Panel& p = *pp;
  p.label = f[0];
  p.bg = f.size() >= 5 && !f[4].empty() ? parse_gradient(f[4]) : bg;
  if (f.size() >= 4 && !f[3].empty())
    for (const std::string& kv : split(f[3], ',')) {
      const auto eq = kv.find('=');
      if (eq == std::string::npos) throw std::invalid_argument("a detail setting is KEY=VALUE: " + kv);
      p.detail.emplace_back(kv.substr(0, eq), std::stof(kv.substr(eq + 1)));
    }
  const auto src = split(f[1], ':');
  if (src[0] == "real" || src[0] == "learned") {
    if (src.size() < 3 || (src[0] == "learned" && src.size() < 4)) throw std::invalid_argument("real:EFFECT:RUN or learned:EFFECT:RUN:FILE: " + f[1]);
    p.kind = src[0];
    if (!sim::parse_effect(src[1], p.effect)) throw std::invalid_argument("no effect " + src[1]);
    p.run0 = std::stoi(src[2]);
    if (p.kind == "learned") p.model = src[3];
    p.size = kSimSize;
    p.rgba.assign(sz(kSimSize) * kSimSize * 4, 0);
    return pp;
  }
  p.kind = "effect";
  p.size = size;
  if (p.detail.empty()) {
    if (nvfx_effect_load(f[1].c_str(), &p.fx) != NVFX_OK) throw std::runtime_error("cannot load " + f[1]);
  } else {
    auto loaded = rollout::load_model(f[1]);
    if (!loaded) throw std::runtime_error(loaded.error());
    set_detail(*loaded, p.detail);
    p.fx = load_effect(*loaded);
  }
  if (nvfx_instance_create(p.fx, size, &p.in) != NVFX_OK) throw std::runtime_error(std::format("cannot play {} at {} px", f[1], size));
  nvfx_instance_set_seed(p.in, seed);
  p.seed = seed;
  nvfx_effect_info info{};
  nvfx_effect_get_info(p.fx, &info);
  p.loops = info.loops != 0;
  p.length = info.fps > 0.f ? static_cast<float>(info.frames) / info.fps : 3.f;
  load_names(p);
  if (f.size() >= 3 && !f[2].empty())
    for (const std::string& k : split(f[2], ';')) {
      const auto c = k.find(':');
      if (c == std::string::npos) throw std::invalid_argument("a key is T:C1,C2,C3: " + k);
      Key key{std::stof(k.substr(0, c)), {}};
      for (const std::string& v : split(k.substr(c + 1), ',')) key.c.push_back(std::stof(v));
      p.keys.push_back(std::move(key));
    }
  if (p.keys.empty()) p.keys.push_back({0.f, std::vector<float>(p.names.size(), 0.5f)});
  std::ranges::sort(p.keys, {}, &Key::t);
  p.rgba.assign(sz(size) * sz(size) * 4, 0);
  return pp;
}

std::vector<float> keyed(const std::vector<Key>& keys, float t) {
  if (t <= keys.front().t) return keys.front().c;
  for (std::size_t i = 1; i < keys.size(); ++i)
    if (t <= keys[i].t) {
      const float u = (t - keys[i - 1].t) / std::max(1e-6f, keys[i].t - keys[i - 1].t);
      std::vector<float> c = keys[i].c;
      for (std::size_t k = 0; k < c.size() && k < keys[i - 1].c.size(); ++k) c[k] = keys[i - 1].c[k] + (keys[i].c[k] - keys[i - 1].c[k]) * u;
      return c;
    }
  return keys.back().c;
}

// Frame f of a panel into p.rgba (premultiplied RGBA).
void render_panel(Panel& p, int f, float restart) {
  const float t = static_cast<float>(f) / kFps;
  if (p.kind == "effect") {
    p.controls = keyed(p.keys, t);
    double at = t;
    if (!p.loops) {  // a one-shot effect: again every period, with the next seed
      const int per = static_cast<int>(std::lround((restart > 0.f ? restart : p.length + 1.f) * kFps));
      if (f / per != p.cycle) {
        p.cycle = f / per;
        nvfx_instance_free(p.in);
        p.in = nullptr;
        if (nvfx_instance_create(p.fx, p.size, &p.in) != NVFX_OK) throw std::runtime_error("cannot create an instance");
        nvfx_instance_set_seed(p.in, p.seed + static_cast<std::uint64_t>(p.cycle));
      }
      at = static_cast<double>(f % per) / kFps;
    }
    nvfx_instance_set_controls(p.in, p.controls.data(), static_cast<int>(p.controls.size()));
    if (nvfx_render(p.in, at, p.rgba.data(), sz(p.size) * 4) != NVFX_OK) throw std::runtime_error("render failed");
    return;
  }
  const float period = restart > 0.f ? restart : sim::effect_loops(p.effect) ? 0.f : static_cast<float>(sim::one_shot_frames(p.effect)) / kFps + 0.5f;
  const int per = period > 0.f ? static_cast<int>(std::lround(period * kFps)) : 1 << 30;
  const int run = p.run0 + f / per, i = f % per;  // i: frames since the start point, less one
  if (run != p.run) start_run(p, run);
  if (p.kind == "real") {
    p.truth->step_frame();
    p.truth->render(p.rgba);
  } else if (nvfx_render(p.in, static_cast<double>(i + 1) / kFps, p.rgba.data(), sz(kSimSize) * 4) != NVFX_OK) {  // frame 0 is the start point
    throw std::runtime_error("render failed");
  }
}

// The panel over its background, scaled to d x d at (x0, y0).
void blit(const Panel& p, Canvas& cv, int x0, int y0, int d, bool nearest) {
  const int n = p.size;
  std::vector<float> img(sz(n) * sz(n) * 3);
  for (int y = 0; y < n; ++y) {
    const Rgb bg = background(y, n, p.bg);
    for (int x = 0; x < n; ++x) {
      const std::size_t i = sz(y) * sz(n) + sz(x);
      const float a = static_cast<float>(p.rgba[i * 4 + 3]) / 255.f;
      for (int k = 0; k < 3; ++k) img[i * 3 + sz(k)] = std::min(255.f, static_cast<float>(p.rgba[i * 4 + sz(k)]) + static_cast<float>(bg[sz(k)]) * (1.f - a));
    }
  }
  const float s = static_cast<float>(n) / static_cast<float>(d);
  for (int y = 0; y < d; ++y)
    for (int x = 0; x < d; ++x) {
      float v[3];
      if (nearest) {
        const int sx = std::min(n - 1, static_cast<int>(static_cast<float>(x) * s)), sy = std::min(n - 1, static_cast<int>(static_cast<float>(y) * s));
        for (int k = 0; k < 3; ++k) v[k] = img[(sz(sy) * sz(n) + sz(sx)) * 3 + sz(k)];
      } else {
        const float fx = std::clamp((static_cast<float>(x) + 0.5f) * s - 0.5f, 0.f, static_cast<float>(n - 1));
        const float fy = std::clamp((static_cast<float>(y) + 0.5f) * s - 0.5f, 0.f, static_cast<float>(n - 1));
        const int ax = static_cast<int>(fx), ay = static_cast<int>(fy), bx = std::min(n - 1, ax + 1), by = std::min(n - 1, ay + 1);
        const float u = fx - static_cast<float>(ax), w = fy - static_cast<float>(ay);
        for (int k = 0; k < 3; ++k) {
          const auto at = [&](int xx, int yy) { return img[(sz(yy) * sz(n) + sz(xx)) * 3 + sz(k)]; };
          v[k] = (at(ax, ay) * (1 - u) + at(bx, ay) * u) * (1 - w) + (at(ax, by) * (1 - u) + at(bx, by) * u) * w;
        }
      }
      const int cx = x0 + x, cy = y0 + y;
      if (cx < 0 || cy < 0 || cx >= cv.w || cy >= cv.h) continue;
      for (int k = 0; k < 3; ++k) cv.rgb[(sz(cy) * sz(cv.w) + sz(cx)) * 3 + sz(k)] = static_cast<std::uint8_t>(std::lround(v[k]));
    }
}

}  // namespace

int main(int argc, char** argv) try {
  std::vector<std::string> specs, captions, notes;
  std::string out, title;
  float seconds = 10.f, zoom = 1.5f, restart = 0.f;
  int size = 256, width = 1280, height = 720, columns = 0, crf = 26;
  std::uint64_t seed = 1;
  Gradient bg = kNight;
  bool timer = false, bars = false, compact = false, nearest = false;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    const auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::invalid_argument(k + " needs a value");
      return argv[++i];
    };
    if (k == "--panel") specs.push_back(next());
    else if (k == "--out") out = next();
    else if (k == "--title") title = next();
    else if (k == "--note") notes.push_back(next());
    else if (k == "--caption") captions.push_back(next());
    else if (k == "--seconds") seconds = std::stof(next());
    else if (k == "--size") size = std::stoi(next());
    else if (k == "--zoom") zoom = std::stof(next());
    else if (k == "--filter") nearest = next() == "nearest";
    else if (k == "--width") width = std::stoi(next());
    else if (k == "--height") height = std::stoi(next());
    else if (k == "--columns") columns = std::stoi(next());
    else if (k == "--crf") crf = std::stoi(next());
    else if (k == "--seed") seed = std::stoull(next());
    else if (k == "--seed-base") g_seed_base = std::stoull(next());
    else if (k == "--restart") restart = std::stof(next());
    else if (k == "--bg") bg = parse_gradient(next());
    else if (k == "--timer") timer = true;
    else if (k == "--bars") bars = true;
    else if (k == "--compact") compact = true;
    else throw std::invalid_argument("unknown option " + k);
  }
  if (out.empty() || specs.empty()) throw std::invalid_argument("nvfx_reel --out F.mp4 --seconds S --panel LABEL|SOURCE[|KEYS[|DETAIL[|BG]]] ... (see the source's header)");
  std::vector<std::unique_ptr<Panel>> panels;
  for (const std::string& s : specs) panels.push_back(make_panel(s, size, seed, bg));

  // Layout: a title bar, a grid of panels (label above, controls below), captions at the bottom.
  const int n = static_cast<int>(panels.size()), cols = columns > 0 ? std::min(columns, n) : n, rows = (n + cols - 1) / cols;
  const int gap = 16, title_h = title.empty() && !timer ? 0 : notes.empty() ? 44 : 42 + 10 * static_cast<int>(notes.size()), label_h = 26, below_h = bars ? 3 * 26 + 8 : compact ? 26 : 3 * 18 + 8;
  const int cap_h = captions.empty() ? 0 : static_cast<int>(captions.size()) * 20 + 12;
  int d0 = 0;
  for (const auto& p : panels) d0 = std::max(d0, static_cast<int>(std::lround(static_cast<float>(p->kind == "effect" ? size : kSimSize) * zoom)));
  const int grid_w = cols * d0 + (cols - 1) * gap, cell_h = label_h + d0 + below_h, grid_h = rows * cell_h + (rows - 1) * gap;
  if (grid_w > width || title_h + grid_h + cap_h > height) throw std::invalid_argument(std::format("{} x {} panels of {} px do not fit {} x {}", cols, rows, d0, width, height));
  const int gx = (width - grid_w) / 2, gy = title_h + (height - title_h - cap_h - grid_h) / 2;
  Canvas cv{width, height, std::vector<std::uint8_t>(sz(width) * sz(height) * 3)};

  const int frames = static_cast<int>(std::lround(seconds * kFps));
  const std::string cmd = std::format("ffmpeg -loglevel error -y -f rawvideo -pix_fmt rgb24 -s {}x{} -r {} -i - -an -c:v libx264 -preset slow -crf {} -threads 2 "
                                      "-pix_fmt yuv420p -movflags +faststart \"{}\"",
                                      width, height, static_cast<int>(kFps), crf, out);
  std::FILE* video = popen(cmd.c_str(), "w");
  if (!video) throw std::runtime_error("cannot start ffmpeg");
  std::println("nvfx_reel: {} panels, {} frames at {} x {} -> {}", n, frames, width, height, out);
  for (int f = 0; f < frames; ++f) {
    cv.fill(0, 0, width, height, {8, 8, 10});
    if (title_h > 0) {
      cv.fill(0, 0, width, title_h, {0, 0, 0});
      cv.text(title, 16, notes.empty() ? 12 : 8, 3, {255, 255, 255});
      for (std::size_t l = 0; l < notes.size(); ++l) cv.text(notes[l], 16, 35 + 10 * static_cast<int>(l), 1, {165, 165, 172});
      if (timer) {  // m:ss.t, from whole tenths of a second (so 59.97 s is 0:59.9, not 0:60.0)
        const int tenths = static_cast<int>(static_cast<float>(f) * 10.f / kFps);
        const std::string ts = std::format("{}:{:02}.{}", tenths / 600, tenths / 10 % 60, tenths % 10);
        cv.text(ts, width - 16 - tools::text_width(ts, 3), notes.empty() ? 12 : 8, 3, {255, 225, 150});
      }
    }
    for (int k = 0; k < n; ++k) {
      Panel& p = *panels[sz(k)];
      render_panel(p, f, restart);
      const int d = static_cast<int>(std::lround(static_cast<float>(p.size) * zoom));
      const int x = gx + (k % cols) * (d0 + gap) + (d0 - d) / 2, y = gy + (k / cols) * (cell_h + gap);
      cv.text(p.label, x, y + 4, 2, {245, 245, 245});
      blit(p, cv, x, y + label_h, d, nearest);
      const int by = y + label_h + d + 8;
      if (compact) {  // "I0.67 W0.24 T0.56", or "I.67 W.24 T.56" where that does not fit the panel
        std::string line;
        const bool tight = d < tools::text_width("I0.00 W0.00 T0.00", 2);
        for (std::size_t c = 0; c < p.controls.size() && c < p.names.size(); ++c) {
          std::string v = std::format("{:.2f}", p.controls[c]);
          if (tight && v.starts_with("0.")) v.erase(0, 1);
          line += std::format("{}{}{}", c ? " " : "", p.names[c].empty() ? '?' : p.names[c][0], v);
        }
        cv.text(line, x, by, 2, {175, 175, 182});
      }
      for (std::size_t c = 0; !compact && c < p.controls.size() && c < p.names.size(); ++c) {
        const float v = std::clamp(p.controls[c], 0.f, 1.f);
        const std::string val = std::format("{:.2f}", p.controls[c]);
        if (bars) {
          const int ry = by + static_cast<int>(c) * 26, lw = 11 * 12, vw = 5 * 12, bw = d - lw - vw - 8;
          cv.text(p.names[c], x, ry + 2, 2, {200, 200, 206});
          cv.fill(x + lw, ry, bw, 18, {40, 40, 46});
          const bool centred = p.names[c] == "wind";
          const int a = x + lw + (centred ? bw / 2 : 0), b = x + lw + static_cast<int>(std::lround(v * static_cast<float>(bw)));
          cv.fill(std::min(a, b), ry + 2, std::max(2, std::abs(b - a)), 14, {255, 170, 70});
          if (centred) cv.fill(x + lw + bw / 2, ry - 2, 1, 22, {150, 150, 156});
          cv.text(val, x + lw + bw + 8, ry + 2, 2, {255, 225, 150});
        } else {
          cv.text(std::format("{} {}", p.names[c], val), x, by + static_cast<int>(c) * 18, 2, {175, 175, 182});
        }
      }
    }
    for (std::size_t c = 0; c < captions.size(); ++c) cv.text(captions[c], 16, height - cap_h + 6 + static_cast<int>(c) * 20, 2, {190, 190, 196});
    if (std::fwrite(cv.rgb.data(), 1, cv.rgb.size(), video) != cv.rgb.size()) throw std::runtime_error("ffmpeg stopped reading");
    if (f % 90 == 0) {
      std::println("  {:.0f} s", static_cast<double>(f) / kFps);
      std::fflush(stdout);
    }
  }
  if (pclose(video) != 0) throw std::runtime_error("ffmpeg failed");
  std::println("nvfx_reel: wrote {}", out);
  return 0;
} catch (const std::exception& e) {
  std::println(stderr, "nvfx_reel: {}", e.what());
  return 1;
}
