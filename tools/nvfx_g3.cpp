// nvfx_g3: study G3 (docs/DCM.md §8), a codec for authored runs built from the learned dynamics, measured as
// rate-distortion curves against flipbooks and video codecs on the same runs.
//
//   probe      one run, a few settings: bytes, quality and time (a sanity check)
//   ladder     the run codec over a ladder of settings, every run of a split; rows appended per run (resumable)
//   baselines  flipbooks (BC3, raw RGBA; as stored and packed by the model-file coder) and video codecs through ffmpeg,
//              every run of a split; rows appended per run (resumable)
//   g3b        G3b: study A's clips, each frame model's output plus a coded residual, against the same baselines
//   summary    curves, the validation-chosen frontier, paired comparisons on test, CSVs and the figures
//   timing     decode time per frame and memory of the run codec and of the video decoders on one pinned core
//   probe-video  one validation run through every video codec (a sanity check)
//
// Splits (docs/DCM.md §4): val = the 10 validation settings of study G with seeds 800000 + i; test = study B's 10 held-out
// settings with seeds 900000 + i (b0..b9), then study D's 8 salt-2 tracking runs (d0..d7; --max-runs 10 leaves them out).
// Runs: 240 frames after a warm-up (explosions: 89 frames after the first), the real run rendered by the simulation's own
// renderer at 128 x 128. Data: $NEURALVFX_DATA (default /root/nvfx-data); the v1 effects from experiments/models/d, rows
// and logs in g3/. The full sequence of commands is in docs/DCM.md §8.
#include "args.hpp"
#include "video_pipe.hpp"

#include <neuralfx/clip.hpp>
#include <neuralfx/cm.hpp>
#include <neuralfx/codec/clip_residual.hpp>
#include <neuralfx/codec/run_codec.hpp>
#include <neuralfx/flipbook.hpp>
#include <neuralfx/metrics.hpp>
#include <neuralfx/nvfx.h>
#include <neuralfx/rollout.hpp>
#include <neuralfx/rollout_train.hpp>
#include <neuralfx/sim.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <mutex>
#include <print>
#include <random>
#include <sched.h>
#include <set>
#include <sstream>
#include <thread>

namespace fs = std::filesystem;
using namespace nfx;

namespace {

constexpr int kSize = 128, kRes = 32;

// The data root: $NEURALVFX_DATA, else /root/nvfx-data. The study's data and logs live in <root>/g3 (outside git).
fs::path data_root() {
  if (const char* d = std::getenv("NEURALVFX_DATA"); d && *d) return d;
  return "/root/nvfx-data";
}
std::string g3_dir() { return (data_root() / "g3").string(); }
std::string models_dir() { return (data_root() / "experiments" / "models" / "d").string(); }
std::string work_dir() { return (data_root() / "g3" / "work").string(); }

std::size_t sz(int v) { return static_cast<std::size_t>(v); }

double thread_seconds() {
  timespec t{};
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
  return static_cast<double>(t.tv_sec) + 1e-9 * static_cast<double>(t.tv_nsec);
}

// --- splits ----------------------------------------------------------------------------------------------------------

using Setting = std::array<float, 3>;

bool off_grid(const Setting& s) {
  const auto off = [](float x, std::initializer_list<float> g) { return std::ranges::all_of(g, [x](float y) { return std::abs(x - y) >= 0.05f; }); };
  return off(s[0], {0.f, 0.5f, 1.f}) && off(s[1], {0.f, 0.25f, 0.5f, 0.75f, 1.f}) && off(s[2], {0.f, 0.5f, 1.f});
}
// Study B's ten held-out settings (nvfx_experiment's b_test_settings).
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
// Study G's ten validation settings (docs/DCM.md §4).
std::vector<Setting> validation_settings() {
  const auto test = test_settings();
  std::mt19937_64 rng(2027);
  std::uniform_real_distribution<float> u(0.05f, 0.95f);
  std::vector<Setting> v;
  while (v.size() < 10) {
    const Setting s{u(rng), u(rng), u(rng)};
    const bool far = std::ranges::all_of(test, [&](const Setting& t) { return std::hypot(s[0] - t[0], s[1] - t[1], s[2] - t[2]) >= 0.05f; });
    if (off_grid(s) && far) v.push_back(s);
  }
  return v;
}

struct RunSpec {
  std::string id;  // "b3", "d5", "v2"
  sim::Params p;
  int warm = 150, frames = 240;
};

bool one_shot(sim::Effect e) { return e == sim::Effect::explosion; }

std::vector<RunSpec> split_runs(sim::Effect e, const std::string& split) {
  std::vector<RunSpec> v;
  const bool ex = one_shot(e);
  const auto setting_run = [&](const std::string& id, const Setting& s, std::uint64_t seed) {
    RunSpec r;
    r.id = id;
    r.p.effect = e;
    r.p.intensity = s[0];
    r.p.wind = s[1];
    r.p.turbulence = s[2];
    r.p.seed = seed;
    r.p.size = kSize;
    r.warm = ex ? 1 : 150;
    r.frames = ex ? 89 : 240;
    return r;
  };
  if (split == "val") {
    const auto s = validation_settings();
    for (int i = 0; i < 10; ++i) v.push_back(setting_run(std::format("v{}", i), s[sz(i)], 800000 + static_cast<std::uint64_t>(i)));
  } else if (split == "test") {
    const auto s = test_settings();
    for (int i = 0; i < 10; ++i) v.push_back(setting_run(std::format("b{}", i), s[sz(i)], 900000 + static_cast<std::uint64_t>(i)));
    for (int i = 0; i < 8; ++i) {  // study D's held-out tracking runs (salt 2), as d-eval
      rollout::SimRecipe r;
      r.effect = e;
      r.salt = 2;
      RunSpec x;
      x.id = std::format("d{}", i);
      x.p = rollout::recipe_run(r, static_cast<std::uint64_t>(i));
      x.warm = ex ? 1 : 100;
      x.frames = ex ? 89 : 240;
      v.push_back(x);
    }
  } else {
    throw std::invalid_argument("unknown split " + split);
  }
  return v;
}

// A simulated run: what the encoder sees and the real frames it is scored against.
struct Simulated {
  codec::TrueRun run;
  Clip ref;  // frames 1..F after the start, rendered by the simulation
};

Simulated simulate(const RunSpec& rs) {
  Simulated out;
  sim::Fluid f(rs.p);
  for (int i = 0; i < rs.warm; ++i) f.step_frame();
  codec::TrueRun& r = out.run;
  r.controls = {rs.p.intensity, rs.p.wind, rs.p.turbulence};
  r.seed = rs.p.seed;
  r.frames = rs.frames;
  r.size = kSize;
  const std::size_t cn = sz(kRes) * kRes * rollout::kPhys, fn = sz(kSize) * kSize;
  r.coarse.resize(sz(rs.frames + 1) * cn);
  r.fine_t.resize(sz(rs.frames + 1) * fn);
  r.fine_d.resize(sz(rs.frames + 1) * fn);
  out.ref.allocate(kSize, rs.frames);
  out.ref.fps = rs.p.fps;
  out.ref.effect = std::string(sim::effect_name(rs.p.effect));
  const auto keep = [&](int i) {
    const sim::State st = f.state();
    if (i == 0) r.time0 = st.time;
    rollout::coarse_from_sim(st, kRes, rs.p.fps, std::span(r.coarse).subspan(sz(i) * cn, cn));
    std::copy(st.temp.begin(), st.temp.end(), r.fine_t.begin() + static_cast<std::ptrdiff_t>(sz(i) * fn));
    std::copy(st.soot.begin(), st.soot.end(), r.fine_d.begin() + static_cast<std::ptrdiff_t>(sz(i) * fn));
  };
  keep(0);
  for (int i = 1; i <= rs.frames; ++i) {
    f.step_frame();
    f.render(out.ref.frame(i - 1));
    keep(i);
  }
  return out;
}

// Active PSNR and PSNR as metrics::score computes them, without SSIM (ssim is reported as -1).
metrics::ClipScores quick_scores(const Clip& ref, std::span<const std::uint8_t> test) {
  double mse_sum = 0, active = 0, n_active = 0;
  const std::size_t fb = ref.frame_bytes();
  for (int f = 0; f < ref.frames; ++f) {
    const auto r = ref.frame(f);
    const auto t = test.subspan(sz(f) * fb, fb);
    double se = 0;
    for (std::size_t i = 0; i < fb; i += 4) {
      double px = 0;
      for (std::size_t c = 0; c < 4; ++c) {
        const double d = (static_cast<double>(r[i + c]) - static_cast<double>(t[i + c])) / 255.0;
        px += d * d;
      }
      se += px;
      if (std::max({r[i], r[i + 1], r[i + 2], r[i + 3]}) > 4 || std::max({t[i], t[i + 1], t[i + 2], t[i + 3]}) > 4) {
        active += px;
        n_active += 4;
      }
    }
    mse_sum += se / static_cast<double>(fb);
  }
  metrics::ClipScores s;
  s.psnr = metrics::psnr_from_mse(mse_sum / ref.frames);
  s.active_psnr = n_active > 0 ? metrics::psnr_from_mse(active / n_active) : metrics::kPsnrCap;
  s.ssim = -1;
  return s;
}

Clip as_clip(const std::vector<std::uint8_t>& rgba, int frames) {
  Clip c;
  c.allocate(kSize, frames);
  std::copy(rgba.begin(), rgba.end(), c.rgba.begin());
  return c;
}

rollout::Model load_effect(const fs::path& models, sim::Effect e) {
  auto m = rollout::load_model(models / std::format("{}.nvfx", sim::effect_name(e)));
  if (!m) throw std::runtime_error(m.error());
  return std::move(*m);
}

std::vector<sim::Effect> effects_of(const std::string& list) {
  std::vector<sim::Effect> v;
  for (const auto e : sim::kEffects) {
    if (list.empty() || list.find(sim::effect_name(e)) != std::string::npos) v.push_back(e);
  }
  return v;
}

// --- CSV helpers ---------------------------------------------------------------------------------------------------------

std::vector<std::map<std::string, std::string>> read_csv(const fs::path& path) {
  std::vector<std::map<std::string, std::string>> rows;
  std::ifstream in(path);
  std::string line;
  if (!std::getline(in, line)) return rows;
  std::vector<std::string> head;
  {
    std::stringstream ss(line);
    std::string c;
    while (std::getline(ss, c, ',')) head.push_back(c);
  }
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::map<std::string, std::string> r;
    std::stringstream ss(line);
    std::string c;
    for (std::size_t i = 0; std::getline(ss, c, ','); ++i) {
      if (i < head.size()) r[head[i]] = c;
    }
    rows.push_back(std::move(r));
  }
  return rows;
}

// Appends rows to a CSV (header written when the file is new), flushed at once: a restarted job resumes from it.
class Appender {
 public:
  Appender(fs::path path, std::string header) : path_(std::move(path)), header_(std::move(header)) {}
  void add(const std::vector<std::string>& rows) {
    const std::lock_guard lock(mu_);
    const bool fresh = !fs::exists(path_);
    fs::create_directories(path_.parent_path());
    std::ofstream o(path_, std::ios::app);
    if (fresh) o << header_ << "\n";
    for (const auto& r : rows) o << r << "\n";
  }
  std::set<std::string> keys(const std::vector<std::string>& cols) const {
    std::set<std::string> k;
    for (const auto& r : read_csv(path_)) {
      std::string key;
      for (const auto& c : cols) key += (r.count(c) ? r.at(c) : "") + "|";
      k.insert(key);
    }
    return k;
  }

 private:
  fs::path path_;
  std::string header_;
  std::mutex mu_;
};

template <class F>
void parallel(int n, int threads, F&& fn) {
  std::atomic<int> next{0};
  std::vector<std::jthread> pool;
  for (int t = 0; t < std::max(1, threads); ++t) {
    pool.emplace_back([&] {
      for (int i; (i = next++) < n;) fn(i);
    });
  }
}

// --- the codec ladder ------------------------------------------------------------------------------------------------------

// The codec's defaults for the study, from probes on validation run v0 of fire: the fine fields start from the coarse
// state (a coded fine start costs about 350 bytes and is forgotten within a second), only heat and soot are corrected
// (velocity doubles the cost for about 1 dB), and a dead-zone quantiser (rounding offset 0.3).
codec::Settings base_settings() {
  codec::Settings s;
  s.k = 8;
  s.q = 0.8f;
  s.q_vel = 0.f;
  s.q_mat = 1.f;
  s.start_nearest = true;
  s.start_fine = 0;
  s.kf = 0;
  s.round = 0.3f;
  s.side_context = false;
  return s;
}

// The validation ladder (every point a setting of the codec; the frontier is chosen from it):
//   core     the start (q_start alone, or none: the nearest stored start as it is) and coarse corrections every k frames
//            at step q;
//   fine     a fine residual at 128 px every frame at step qf (no coarse correction: the coarse heat and soot follow it),
//            and two with velocity corrections every frame;
//   variants one change at a time at two core points (k8 q0.8, k4 q0.4) and one fine point (qf 0.07).
std::vector<codec::Settings> ladder_set(const std::string& name) {
  std::vector<codec::Settings> v;
  const auto with = [](codec::Settings s, auto f) {
    f(s);
    return s;
  };
  const auto core = [&](int k, float q, float s0 = 0.f) {
    return with(base_settings(), [&](codec::Settings& x) {
      x.k = k;
      x.q = q;
      x.q_start = s0;
    });
  };
  const auto fine = [&](float qf) {
    return with(base_settings(), [&](codec::Settings& x) {
      x.k = 0;
      x.q = 0.4f;
      x.kf = 1;
      x.fine_res = 128;
      x.qf = qf;
    });
  };
  const bool all = name == "val" || name == "all";
  if (name == "novar") {  // smoke and explosion: the variants were measured on fire only (CPU budget)
    for (const auto* part : {"core", "fine", "extra"}) {
      const auto p = ladder_set(part);
      v.insert(v.end(), p.begin(), p.end());
    }
    return v;
  }
  if (name == "core" || all) {
    for (const float s0 : {60.f, 1.6f, 0.8f, 0.4f, 0.2f}) v.push_back(core(0, 0.4f, s0));
    for (const int k : {2, 4, 8, 16, 32}) {
      for (const float q : {0.2f, 0.4f, 0.8f, 1.6f}) v.push_back(core(k, q));
    }
    for (const int k : {8, 16, 32}) v.push_back(core(k, 0.8f, 60.f));
  }
  if (name == "fine" || all) {
    for (const float qf : {0.025f, 0.035f, 0.05f, 0.07f, 0.1f, 0.14f, 0.2f}) v.push_back(fine(qf));
    for (const float qf : {0.05f, 0.1f}) {
      v.push_back(with(fine(qf), [](codec::Settings& x) {
        x.k = 1;
        x.q = 0.1f;
        x.q_vel = 1.f;
        x.q_mat = 0.f;
      }));
    }
  }
  if (name == "variants" || all) {
    for (const auto& b : {core(8, 0.8f), core(4, 0.4f)}) {
      v.push_back(with(b, [](codec::Settings& x) { x.round = 0.5f; }));
      v.push_back(with(b, [](codec::Settings& x) { x.q_vel = 1.f; }));
      v.push_back(with(b, [](codec::Settings& x) { x.start_fine = 64; x.q_fine_start = 0.05f; }));
      v.push_back(with(b, [](codec::Settings& x) { x.start_nearest = false; }));
      v.push_back(with(b, [](codec::Settings& x) { x.side_context = true; }));
    }
    const auto b = fine(0.07f);
    v.push_back(with(b, [](codec::Settings& x) { x.round = 0.5f; }));
    v.push_back(with(b, [](codec::Settings& x) { x.side_context = true; }));
    v.push_back(with(b, [](codec::Settings& x) { x.fine_sets_coarse = false; }));
    v.push_back(with(b, [](codec::Settings& x) { x.fine_res = 64; }));
  }
  // added after fire's first validation runs put the frontier at the grid's edge (q 1.6, k 2) and left a gap between
  // the coarse points (at most about 21 dB) and the fine ones (from about 24 dB)
  if (name == "extra" || all) {
    for (const float q : {0.8f, 1.6f, 3.2f}) v.push_back(core(1, q));
    for (const int k : {2, 4, 8}) v.push_back(core(k, 3.2f));
    for (const float qf : {0.1f, 0.14f, 0.2f, 0.3f}) v.push_back(with(fine(qf), [](codec::Settings& x) { x.fine_res = 64; }));
    for (const float qf : {0.3f, 0.4f}) v.push_back(fine(qf));
  }
  if (v.empty()) throw std::invalid_argument("unknown ladder set " + name);
  return v;
}

// Settings from a description ("k4_q0.1000_..."): the frontier chosen on validation is replayed on test by name.
codec::Settings parse_settings(const std::string& d) {
  codec::Settings s = base_settings();
  std::stringstream ss(d);
  std::string tok;
  while (std::getline(ss, tok, '_')) {
    if (tok.empty()) continue;
    const auto num = [&](std::size_t from) { return tok.substr(from); };
    if (tok == "noflow") s.correct_flow = false;
    else if (tok == "nosync") s.fine_sets_coarse = false;
    else if (tok == "side") s.side_context = true;
    else if (tok[0] == 'o') s.round = std::stof(num(1));
    else if (tok[0] == 'm') s.q_mat = std::stof(num(1));
    else if (tok.rfind("kf", 0) == 0) s.kf = std::stoi(num(2));
    else if (tok.rfind("qf", 0) == 0) s.qf = std::stof(num(2));
    else if (tok.rfind("fs", 0) == 0) s.q_fine_start = std::stof(num(2));
    else if (tok[0] == 'k') s.k = std::stoi(num(1));
    else if (tok[0] == 'q') s.q = std::stof(num(1));
    else if (tok[0] == 'v') s.q_vel = std::stof(num(1));
    else if (tok[0] == 's') {
      const char last = tok.back();
      s.start_nearest = last == 'n';
      s.q_start = std::stof(tok.substr(1, tok.size() - 2));
    } else if (tok[0] == 'f') s.start_fine = std::stoi(num(1));
    else if (tok[0] == 'r') s.fine_res = std::stoi(num(1));
    else throw std::invalid_argument("cannot parse settings " + d);
  }
  return s;
}

const char* kLadderHeader =
    "effect,split,run,settings,bytes,header_bytes,coarse_start_bytes,fine_start_bytes,coarse_bytes,fine_bytes,coarse_planes,fine_planes,frames,"
    "active_psnr,psnr,ssim,encode_s";

std::string ladder_row(sim::Effect e, const std::string& split, const std::string& run, const codec::Settings& s, const codec::Encoded& enc,
                       const metrics::ClipScores& sc, int frames, double secs) {
  return std::format("{},{},{},{},{},{},{:.1f},{:.1f},{:.1f},{:.1f},{},{},{},{:.4f},{:.4f},{:.5f},{:.2f}", sim::effect_name(e), split, run,
                     codec::describe(s), enc.stream.size(), enc.header_bytes, enc.coarse_start_bytes, enc.fine_start_bytes, enc.coarse_bytes,
                     enc.fine_bytes, enc.coarse_planes, enc.fine_planes, frames, sc.active_psnr, sc.psnr, sc.ssim, secs);
}

void cmd_ladder(const tools::Args& a) {
  const fs::path models = a.str("models", models_dir());
  const fs::path out = a.str("out", g3_dir());
  const std::string split = a.str("split", "val"), set = a.str("set", "val");
  const int threads = a.i("threads", 1), max_runs = a.i("max-runs", 1000);
  const bool ssim = !a.flag("no-ssim");  // validation chooses by active PSNR only
  std::vector<codec::Settings> points;
  if (a.has("points")) {  // a file of settings names, one per line (the validation frontier)
    std::ifstream in(a.str("points"));
    for (std::string l; std::getline(in, l);) {
      if (!l.empty()) points.push_back(parse_settings(l));
    }
  } else {
    points = ladder_set(set);
  }
  for (const auto e : effects_of(a.str("effects"))) {
    const rollout::Model M = load_effect(models, e);
    auto runs = split_runs(e, split);
    if (static_cast<int>(runs.size()) > max_runs) runs.resize(sz(max_runs));
    Appender app(out / std::format("ladder_{}_{}.csv", split, sim::effect_name(e)), kLadderHeader);
    const auto done = app.keys({"run", "settings"});
    parallel(static_cast<int>(runs.size()), threads, [&](int ri) {
      const RunSpec& rs = runs[sz(ri)];
      std::vector<const codec::Settings*> todo;
      for (const auto& s : points) {
        if (!done.count(rs.id + "|" + codec::describe(s) + "|")) todo.push_back(&s);
      }
      if (todo.empty()) return;
      const Simulated S = simulate(rs);
      bool checked = false;
      for (const codec::Settings* s : todo) {
        const double t0 = thread_seconds();
        const codec::Encoded enc = codec::encode(M, S.run, *s, true);
        const double secs = thread_seconds() - t0;
        if (!checked) {  // the decoder reproduces the encoder's reconstruction (once per run; the tests check more)
          const auto dec = codec::decode(M, enc.stream);
          if (!dec || !dec->verified || dec->rgba != enc.frames) throw std::runtime_error("decoder mismatch on " + rs.id + " " + codec::describe(*s));
          checked = true;
        }
        const auto sc = ssim ? metrics::score(S.ref, as_clip(enc.frames, rs.frames)) : quick_scores(S.ref, enc.frames);
        app.add({ladder_row(e, split, rs.id, *s, enc, sc, rs.frames, secs)});
      }
      std::println("ladder {} {} {}: {} points", sim::effect_name(e), split, rs.id, todo.size());
      std::fflush(stdout);
    });
  }
}

// --- baselines -------------------------------------------------------------------------------------------------------------

const char* kBaseHeader = "effect,split,run,method,config,bytes,file_bytes,frames,active_psnr,psnr,ssim,encode_s,decode_s";

// Box-downsample every frame of a clip to res (a frame size divisible by res).
std::vector<std::uint8_t> downsample_clip(const Clip& c, int res) {
  std::vector<std::uint8_t> out;
  out.reserve(sz(c.frames) * sz(res) * sz(res) * 4);
  for (int f = 0; f < c.frames; ++f) {
    const auto d = flipbook::downsample(c.frame(f), c.size, res);
    out.insert(out.end(), d.begin(), d.end());
  }
  return out;
}

// Bilinear upsampling of frames at res to the clip size (cell centres aligned), as a game samples a smaller texture.
Clip upsample_clip(std::span<const std::uint8_t> rgba, int res, int frames) {
  Clip c;
  c.allocate(kSize, frames);
  const float k = static_cast<float>(res) / static_cast<float>(kSize);
  std::array<float, 4> px{};
  const std::size_t fb = sz(res) * sz(res) * 4;
  for (int f = 0; f < frames; ++f) {
    const auto src = rgba.subspan(sz(f) * fb, fb);
    auto dst = c.frame(f);
    for (int y = 0; y < kSize; ++y) {
      for (int x = 0; x < kSize; ++x) {
        flipbook::sample_bilinear(src, res, (static_cast<float>(x) + 0.5f) * k - 0.5f, (static_cast<float>(y) + 0.5f) * k - 0.5f, px);
        for (std::size_t ch = 0; ch < 4; ++ch) dst[(sz(y) * kSize + sz(x)) * 4 + ch] = static_cast<std::uint8_t>(std::clamp(px[ch] + 0.5f, 0.f, 255.f));
      }
    }
  }
  return c;
}

// The stored form of a flipbook as tensors for the model-file coder (as nvfx_pack measures flipbooks).
std::vector<cm::Tensor> flipbook_tensors(const Clip& ref, const flipbook::Spec& spec, const flipbook::Flipbook& fb) {
  const auto F = static_cast<std::uint32_t>(fb.kept.size()), R = static_cast<std::uint32_t>(spec.res);
  cm::Tensor img;
  img.shape.width = 1;
  img.shape.channels = true;
  for (const int f : fb.kept) {
    auto small = flipbook::downsample(ref.frame(f), ref.size, spec.res);
    if (spec.codec == flipbook::Codec::bc3) small = flipbook::compress_bc3(small, spec.res, spec.res);
    img.values.insert(img.values.end(), small.begin(), small.end());
  }
  if (spec.codec == flipbook::Codec::bc3) {
    img.shape.kind = cm::Kind::bc3;
    img.shape.dims = {F, R / 4, R / 4, 16};
  } else {
    img.shape.kind = cm::Kind::rgba;
    img.shape.dims = {F, R, R, 4};
  }
  return {std::move(img)};
}

// Flipbooks of the run: kept frames F, F/2, F/4, F/8, F/16 at 128, 64 and 32 px in BC3, and at 64 and 32 px raw.
std::vector<flipbook::Spec> flipbook_specs(int frames) {
  std::vector<flipbook::Spec> v;
  std::vector<int> keep;
  for (int k = frames; k >= 4 && keep.size() < 5; k /= 2) keep.push_back(k);
  for (const int res : {128, 64, 32}) {
    for (const int k : keep) v.push_back({k, res, flipbook::Codec::bc3, 0});
  }
  for (const int res : {64, 32}) {
    for (const int k : keep) v.push_back({k, res, flipbook::Codec::raw, 0});
  }
  return v;
}

struct VideoConfig {
  video::Codec codec;
  int res = kSize;  // coded at this side, upsampled to 128 for scoring
  std::string name() const { return res == kSize ? codec.name : std::format("{}_{}px", codec.name, res); }
};

std::vector<VideoConfig> video_configs(const std::string& list) {
  std::vector<VideoConfig> v;
  const auto codecs = video::study_codecs();
  std::stringstream ss(list);
  for (std::string t; std::getline(ss, t, ',');) {
    if (t.empty()) continue;
    std::vector<int> rungs;  // "name:q1/q2/...": only these rungs of the codec's ladder
    if (const auto c = t.find(':'); c != std::string::npos) {
      std::stringstream qs(t.substr(c + 1));
      for (std::string q; std::getline(qs, q, '/');) rungs.push_back(std::stoi(q));
      t = t.substr(0, c);
    }
    int res = kSize;
    std::string name = t;
    for (const int r : {64, 32}) {
      if (const auto p = t.find(std::format("_{}px", r)); p != std::string::npos) {
        res = r;
        name = t.substr(0, p);
      }
    }
    const auto it = std::ranges::find_if(codecs, [&](const video::Codec& c) { return c.name == name; });
    if (it == codecs.end()) throw std::invalid_argument("unknown codec " + t);
    v.push_back({*it, res});
    if (!rungs.empty()) v.back().codec.ladder = rungs;
  }
  return v;
}

void cmd_baselines(const tools::Args& a) {
  const fs::path out = a.str("out", g3_dir());
  const fs::path work = a.str("work", work_dir());
  const std::string split = a.str("split", "test"), what = a.str("what", "flipbook");
  const std::string runs_only = a.str("runs");  // e.g. "v0,v1" (a subset, for the choice of video formats)
  const auto vcfg = video_configs(a.str("codecs", "x264,x265,vp9,vp9a,aom,svt"));
  const bool keep_files = a.flag("keep");
  const int max_runs = a.i("max-runs", 1000);
  const std::size_t pack_max = static_cast<std::size_t>(a.i("pack-max-kb", 1024)) * 1024;  // larger flipbooks: stored size only
  for (const auto e : effects_of(a.str("effects"))) {
    auto runs = split_runs(e, split);
    if (static_cast<int>(runs.size()) > max_runs) runs.resize(sz(max_runs));
    Appender app(out / std::format("base_{}_{}.csv", split, sim::effect_name(e)), kBaseHeader);
    const auto done = app.keys({"run", "method", "config"});
    for (const RunSpec& rs : runs) {
      if (!runs_only.empty() && (std::format(",{},", runs_only)).find("," + rs.id + ",") == std::string::npos) continue;
      // what is missing for this run
      std::vector<flipbook::Spec> specs;
      std::vector<std::pair<const VideoConfig*, int>> vids;
      if (what == "flipbook") {
        for (const auto& sp : flipbook_specs(rs.frames)) {
          if (!done.count(rs.id + "|flipbook_" + std::string(sp.codec == flipbook::Codec::bc3 ? "bc3" : "raw") + "|" + sp.describe() + "|")) specs.push_back(sp);
        }
      } else {
        for (const auto& vc : vcfg) {
          for (const int q : vc.codec.ladder) {
            if (!done.count(rs.id + "|" + vc.name() + "|" + std::to_string(q) + "|")) vids.emplace_back(&vc, q);
          }
        }
      }
      if (specs.empty() && vids.empty()) continue;
      const Simulated S = simulate(rs);
      const std::string en(sim::effect_name(e));
      for (const auto& sp : specs) {
        const flipbook::Flipbook fb = flipbook::build(S.ref, sp);
        const Clip played = flipbook::play(fb);
        const auto sc = metrics::score(S.ref, played);
        const auto tensors = flipbook_tensors(S.ref, sp, fb);
        std::size_t stored = 0;
        for (const auto& t : tensors) stored += t.values.size();
        if (stored != fb.bytes) throw std::runtime_error("flipbook: stored size differs from flipbook::memory_bytes");
        const std::string kind = sp.codec == flipbook::Codec::bc3 ? "bc3" : "raw";
        const std::string row_stored = std::format("{},{},{},flipbook_{},{},{},{},{},{:.4f},{:.4f},{:.5f},0,0", en, split, rs.id, kind, sp.describe(), stored, stored,
                                                   rs.frames, sc.active_psnr, sc.psnr, sc.ssim);
        if (stored > pack_max) {
          app.add({row_stored});
          continue;
        }
        const double t0 = thread_seconds();
        const cm::Packed p = cm::pack_tensors(tensors);
        const double te = thread_seconds() - t0;
        const double t1 = thread_seconds();
        const auto back = cm::unpack_tensors(p.data);
        const double td = thread_seconds() - t1;
        if (!back || (*back)[0].values != tensors[0].values) throw std::runtime_error("flipbook: packing round trip failed");
        app.add({row_stored,
                 std::format("{},{},{},flipbook_{}_packed,{},{},{},{},{:.4f},{:.4f},{:.5f},{:.3f},{:.3f}", en, split, rs.id, kind, sp.describe(), p.data.size(),
                             p.data.size(), rs.frames, sc.active_psnr, sc.psnr, sc.ssim, te, td)});
      }
      for (const auto& [vc, q] : vids) {
        std::vector<std::uint8_t> src;
        std::span<const std::uint8_t> in = S.ref.rgba;
        if (vc->res != kSize) {
          src = downsample_clip(S.ref, vc->res);
          in = src;
        }
        const auto r = video::roundtrip(vc->codec, q, in, vc->res, rs.frames, rs.p.fps, work, std::format("{}_{}_{}", en, split, rs.id));
        if (!r.ok) throw std::runtime_error(std::format("{} q{} on {}: {}", vc->name(), q, rs.id, r.error));
        const Clip dec = vc->res == kSize ? as_clip(r.rgba, rs.frames) : upsample_clip(r.rgba, vc->res, rs.frames);
        const auto sc = metrics::score(S.ref, dec);
        app.add({std::format("{},{},{},{},{},{},{},{},{:.4f},{:.4f},{:.5f},{:.3f},{:.3f}", en, split, rs.id, vc->name(), q, r.bytes, r.file_bytes, rs.frames,
                             sc.active_psnr, sc.psnr, sc.ssim, r.encode_cpu_s, r.decode_cpu_s)});
        if (!keep_files) {
          std::error_code ec;
          fs::remove(work / std::format("{}_{}_{}_{}_q{}.{}", en, split, rs.id, vc->codec.name, q, vc->codec.ext), ec);
        }
      }
      std::println("baselines {} {} {}: {} flipbooks, {} videos", en, split, rs.id, specs.size(), vids.size());
      std::fflush(stdout);
    }
  }
}

// --- G3b: study A clips, a frame model plus a coded residual ----------------------------------------------------------------

const char* kG3bHeader = "clip,method,config,bytes,model_bytes,frames,active_psnr,psnr,ssim";

std::vector<std::string> a_clip_names() {
  std::vector<std::string> v;
  for (const auto e : sim::kEffects) {
    for (int k = 0; k < 4; ++k) v.push_back(std::format("{}_{}", sim::effect_name(e), k));
  }
  return v;
}

// The frame model's clip through the runtime, as study A scores it (variation 0, no drift, frame f at f / fps).
Clip model_clip(const fs::path& model, int frames) {
  nvfx_effect* fx = nullptr;
  if (nvfx_effect_load(model.string().c_str(), &fx) != NVFX_OK) throw std::runtime_error("cannot load " + model.string());
  nvfx_effect_info info{};
  nvfx_effect_get_info(fx, &info);
  nvfx_instance* in = nullptr;
  if (nvfx_instance_create(fx, kSize, &in) != NVFX_OK) throw std::runtime_error("instance");
  nvfx_instance_set_drift(in, 0.f);
  nvfx_instance_set_variation(in, 0);
  Clip c;
  c.allocate(kSize, frames);
  for (int f = 0; f < frames; ++f) nvfx_render(in, f / static_cast<double>(info.fps), c.frame(f).data(), kSize * 4);
  nvfx_instance_free(in);
  nvfx_effect_free(fx);
  return c;
}

void cmd_g3b(const tools::Args& a) {
  const fs::path exp = a.str("experiments", (data_root() / "experiments").string());
  const fs::path out = a.str("out", g3_dir());
  const fs::path work = a.str("work", work_dir());
  const auto vcfg = video_configs(a.str("codecs", "x264_rgb,x265_444,vp9a,aom_444,svt"));
  const float round = a.f("round", 0.3f);
  const std::string model_cfg = a.str("model", "grid_m8");
  Appender app(out / "g3b.csv", kG3bHeader);
  const auto done = app.keys({"clip", "method", "config"});
  for (const auto& name : a_clip_names()) {
    auto ref = read_clip(exp / "clips" / "a" / std::format("{}.nfxclip", name));
    if (!ref) throw std::runtime_error(ref.error());
    const int F = ref->frames;
    const fs::path mpath = exp / "models" / "a" / std::format("{}_{}.nvfx", name, model_cfg);
    std::vector<std::uint8_t> file;
    {
      std::ifstream in(mpath, std::ios::binary);
      file.assign(std::istreambuf_iterator<char>(in), {});
    }
    const std::size_t model_raw = file.size(), model_packed = cm::pack_model(file).data.size();
    const Clip pred = model_clip(mpath, F);
    std::vector<std::string> rows;
    const auto add = [&](const std::string& method, const std::string& config, std::size_t bytes, std::size_t mb, const Clip& c) {
      const auto sc = metrics::score(*ref, c);
      rows.push_back(std::format("{},{},{},{},{},{},{:.4f},{:.4f},{:.5f}", name, method, config, bytes, mb, F, sc.active_psnr, sc.psnr, sc.ssim));
    };
    if (!done.count(name + "|model|" + model_cfg + "|")) {
      add("model", model_cfg, model_raw, model_raw, pred);
      add("model_packed", model_cfg, model_packed, model_packed, pred);
    }
    for (const float step : {2.f, 3.f, 4.f, 6.f, 8.f, 12.f, 16.f, 24.f, 32.f, 48.f}) {
      const std::string cfg = std::format("{}_step{:g}", model_cfg, step);
      if (done.count(name + "|g3b|" + cfg + "|")) continue;
      const auto e = codec::encode_clip_residual(pred.rgba, ref->rgba, kSize, F, step, round);
      const auto d = codec::decode_clip_residual(pred.rgba, e.stream);
      if (!d || *d != e.frames) throw std::runtime_error("g3b: decoder mismatch");
      add("g3b", cfg, e.stream.size() + model_packed, model_packed, as_clip(e.frames, F));
      add("residual_only", cfg, e.stream.size(), 0, as_clip(e.frames, F));
    }
    for (const auto& sp : flipbook::ladder(kSize, F)) {
      if (sp.flow_res > 0) continue;
      const std::string kind = sp.codec == flipbook::Codec::bc3 ? "bc3" : "raw";
      if (done.count(name + "|flipbook_" + kind + "_packed|" + sp.describe() + "|")) continue;
      const flipbook::Flipbook fb = flipbook::build(*ref, sp);
      const Clip played = flipbook::play(fb);
      const cm::Packed p = cm::pack_tensors(flipbook_tensors(*ref, sp, fb));
      add("flipbook_" + kind, sp.describe(), fb.bytes, 0, played);
      add("flipbook_" + kind + "_packed", sp.describe(), p.data.size(), 0, played);
    }
    for (const auto& vc : vcfg) {
      for (const int q : vc.codec.ladder) {
        if (done.count(name + "|" + vc.name() + "|" + std::to_string(q) + "|")) continue;
        const auto r = video::roundtrip(vc.codec, q, ref->rgba, kSize, F, ref->fps, work, "g3b_" + name);
        if (!r.ok) throw std::runtime_error(r.error);
        add(vc.name(), std::to_string(q), r.bytes, 0, as_clip(r.rgba, F));
        std::error_code ec;
        fs::remove(work / std::format("g3b_{}_{}_q{}.{}", name, vc.codec.name, q, vc.codec.ext), ec);
      }
    }
    app.add(rows);
    std::println("g3b {}: {} rows", name, rows.size());
    std::fflush(stdout);
  }
}

// --- summary -----------------------------------------------------------------------------------------------------------------

struct Point {
  std::string run, method, config;
  double bytes = 0, apsnr = 0, psnr = 0, ssim = 0;
};

// Rows of a split (a restarted job can repeat a row: the first of each run, method and config is kept).
std::vector<Point> load_points(const fs::path& dir, const std::string& split, sim::Effect e) {
  std::vector<Point> v;
  std::set<std::string> seen;
  const auto keep = [&](Point p) {
    if (seen.insert(p.run + "|" + p.method + "|" + p.config).second) v.push_back(std::move(p));
  };
  const std::string en(sim::effect_name(e));
  for (const auto& r : read_csv(dir / std::format("ladder_{}_{}.csv", split, en))) {
    keep({r.at("run"), "g3a", r.at("settings"), std::stod(r.at("bytes")), std::stod(r.at("active_psnr")), std::stod(r.at("psnr")), std::stod(r.at("ssim"))});
  }
  for (const auto& r : read_csv(dir / std::format("base_{}_{}.csv", split, en))) {
    keep({r.at("run"), r.at("method"), r.at("config"), std::stod(r.at("bytes")), std::stod(r.at("active_psnr")), std::stod(r.at("psnr")), std::stod(r.at("ssim"))});
  }
  return v;
}

// Mean over runs of each (method, config), only configs present on every run of the split.
struct Mean {
  std::string method, config;
  double bytes = 0, apsnr = 0, psnr = 0, ssim = 0;
  int runs = 0;
};
std::vector<Mean> means(const std::vector<Point>& pts, int n_runs) {
  std::map<std::pair<std::string, std::string>, Mean> m;
  for (const auto& p : pts) {
    Mean& x = m[{p.method, p.config}];
    x.method = p.method;
    x.config = p.config;
    x.bytes += p.bytes;
    x.apsnr += p.apsnr;
    x.psnr += p.psnr;
    x.ssim += p.ssim;
    ++x.runs;
  }
  std::vector<Mean> v;
  for (auto& [k, x] : m) {
    if (x.runs != n_runs) continue;
    const double n = x.runs;
    x.bytes /= n;
    x.apsnr /= n;
    x.psnr /= n;
    x.ssim /= n;
    v.push_back(x);
  }
  return v;
}

// The configs of a method on its frontier of means: no other config has fewer mean bytes and a higher mean active PSNR.
// The codec's side-context option is left out (the codec as shipped decodes on any build; its gain is reported apart).
std::vector<std::string> frontier(const std::vector<Mean>& ms, const std::string& method) {
  std::vector<Mean> v;
  for (const auto& m : ms) {
    if (m.method == method && m.config.find("_side") == std::string::npos) v.push_back(m);
  }
  std::ranges::sort(v, [](const Mean& a, const Mean& b) { return a.bytes < b.bytes || (a.bytes == b.bytes && a.apsnr > b.apsnr); });
  std::vector<std::string> out;
  double best = -1e9;
  for (const auto& m : v) {
    if (m.apsnr > best + 1e-9) {
      out.push_back(m.config);
      best = m.apsnr;
    }
  }
  return out;
}

// One run's curve of a family: its configs' (bytes, quality), sorted by bytes, quality made non-decreasing (the best
// quality at or below each size: a user picks the best setting that fits).
struct Curve {
  std::vector<double> lb, q;  // log bytes, quality
  bool empty() const { return lb.empty(); }
};
Curve curve(const std::vector<Point>& pts, const std::string& run, const std::string& method, const std::set<std::string>& configs, int metric) {
  std::vector<std::pair<double, double>> v;
  for (const auto& p : pts) {
    if (p.run != run || p.method != method || !configs.count(p.config)) continue;
    v.emplace_back(std::log(p.bytes), metric == 0 ? p.apsnr : metric == 1 ? p.psnr : p.ssim);
  }
  std::ranges::sort(v);
  Curve c;
  double best = -1e9;
  for (const auto& [b, q] : v) {
    best = std::max(best, q);
    c.lb.push_back(b);
    c.q.push_back(best);
  }
  return c;
}
// log bytes needed to reach quality Q (NaN if never reached); `floor` is set when the smallest config already exceeds Q.
double bytes_at(const Curve& c, double Q, bool& floor) {
  floor = false;
  if (c.empty() || Q > c.q.back()) return std::nan("");
  if (Q <= c.q.front()) {
    floor = true;
    return c.lb.front();
  }
  for (std::size_t i = 1; i < c.q.size(); ++i) {
    if (c.q[i] >= Q) {
      if (c.q[i] == c.q[i - 1]) return c.lb[i];
      const double t = (Q - c.q[i - 1]) / (c.q[i] - c.q[i - 1]);
      return c.lb[i - 1] + t * (c.lb[i] - c.lb[i - 1]);
    }
  }
  return std::nan("");
}
// quality at a byte budget (NaN below the smallest config; the best quality above the largest).
double quality_at(const Curve& c, double lb) {
  if (c.empty() || lb < c.lb.front()) return std::nan("");
  for (std::size_t i = c.lb.size(); i-- > 0;) {
    if (c.lb[i] <= lb) {
      if (i + 1 == c.lb.size()) return c.q[i];
      const double t = (lb - c.lb[i]) / (c.lb[i + 1] - c.lb[i]);
      return c.q[i] + t * (c.q[i + 1] - c.q[i]);
    }
  }
  return std::nan("");
}

// --- the figure ----------------------------------------------------------------------------------------------------------

struct Series {
  std::string label;
  std::vector<std::pair<double, double>> pts;  // (bytes, active PSNR), sorted by bytes
  bool hero = false, dashed = false;
};

// Rate-distortion curves as small multiples (one panel per effect, one log-scale axis each), light surface, the default
// categorical palette in fixed order (docs: dataviz reference palette), a legend and an end label per series.
void write_svg(const fs::path& path, const std::vector<std::pair<std::string, std::vector<Series>>>& panels, const std::string& title, const std::string& note,
               double y1 = 50, const std::string& unit = "run") {
  static const char* colors[8] = {"#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"};
  const auto np = static_cast<double>(panels.size());
  const double pw = panels.size() == 1 ? 640 : 380, ph = 330, ml = 52, mr = 14, gap = 24, mb = 64;
  const double W = ml + np * pw + (np - 1) * gap + mr;
  // legend entries laid out in rows that fit the width; the note wrapped at spaces
  const auto& lser = panels.front().second;
  std::vector<std::pair<double, double>> lpos;  // x, y of each legend entry
  {
    double lx = ml, ly = 64;
    for (std::size_t i = 0; i < lser.size() && i < 8; ++i) {
      const double w = 24 + 6.6 * static_cast<double>(lser[i].label.size()) + 18;
      if (lx + w > W - mr && lx > ml) {
        lx = ml;
        ly += 20;
      }
      lpos.emplace_back(lx, ly);
      lx += w;
    }
  }
  std::vector<std::string> note_lines;
  {
    const auto per_line = static_cast<std::size_t>((W - ml - mr) / 6.2);
    std::string rest = note;
    while (rest.size() > per_line) {
      std::size_t cut = rest.rfind(' ', per_line);
      if (cut == std::string::npos) cut = per_line;
      note_lines.push_back(rest.substr(0, cut));
      rest = rest.substr(cut + 1);
    }
    note_lines.push_back(rest);
  }
  const double note_h = 16.0 * static_cast<double>(note_lines.size() - 1);
  for (auto& [x, y] : lpos) y += note_h;
  const double mt = (lpos.empty() ? 64 + note_h : lpos.back().second) + 36;
  const double H = mt + ph + mb;
  double bmin = 1e12, bmax = 1;
  for (const auto& pnl : panels) {
    for (const auto& ser : pnl.second) {
      for (const auto& pt : ser.pts) {
        bmin = std::min(bmin, pt.first);
        bmax = std::max(bmax, pt.first);
      }
    }
  }
  const double x0 = std::log10(std::min(30.0, 0.7 * bmin)), x1 = std::log10(std::max(1e4, 1.5 * bmax)), y0 = 10;
  std::string o = std::format("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{:.0f}\" height=\"{:.0f}\" viewBox=\"0 0 {:.0f} {:.0f}\" "
                              "font-family=\"system-ui, -apple-system, Segoe UI, Helvetica, Arial, sans-serif\">\n",
                              W, H, W, H);
  o += std::format("<rect width=\"{:.0f}\" height=\"{:.0f}\" fill=\"#fcfcfb\"/>\n", W, H);
  o += std::format("<text x=\"{}\" y=\"24\" font-size=\"16\" font-weight=\"600\" fill=\"#0b0b0b\">{}</text>\n", ml, title);
  for (std::size_t i = 0; i < note_lines.size(); ++i) {
    o += std::format("<text x=\"{}\" y=\"{:.0f}\" font-size=\"12\" fill=\"#52514e\">{}</text>\n", ml, 44.0 + 16.0 * static_cast<double>(i), note_lines[i]);
  }
  // legend, from the first panel's series order
  for (std::size_t i = 0; i < lpos.size(); ++i) {
    const auto [lx, ly] = lpos[i];
    o += std::format("<line x1=\"{:.1f}\" y1=\"{:.1f}\" x2=\"{:.1f}\" y2=\"{:.1f}\" stroke=\"{}\" stroke-width=\"{}\"{}/>\n", lx, ly, lx + 18, ly, colors[i],
                     lser[i].hero ? 3 : 2, lser[i].dashed ? " stroke-dasharray=\"5 3\"" : "");
    o += std::format("<circle cx=\"{:.1f}\" cy=\"{:.1f}\" r=\"4\" fill=\"{}\" stroke=\"#fcfcfb\" stroke-width=\"2\"/>\n", lx + 9, ly, colors[i]);
    o += std::format("<text x=\"{:.1f}\" y=\"{:.1f}\" font-size=\"12\" fill=\"#0b0b0b\">{}</text>\n", lx + 24, ly + 4, lser[i].label);
  }
  for (std::size_t p = 0; p < panels.size(); ++p) {
    const double px = ml + static_cast<double>(p) * (pw + gap);
    const auto X = [&](double b) { return px + (std::log10(b) - x0) / (x1 - x0) * pw; };
    const auto Y = [&](double q) { return mt + ph - (std::clamp(q, y0, y1) - y0) / (y1 - y0) * ph; };
    o += std::format("<text x=\"{:.1f}\" y=\"{:.1f}\" font-size=\"14\" font-weight=\"600\" fill=\"#0b0b0b\">{}</text>\n", px, mt - 10, panels[p].first);
    for (double q = y0; q <= y1 + 1e-9; q += 5) {
      o += std::format("<line x1=\"{:.1f}\" y1=\"{:.1f}\" x2=\"{:.1f}\" y2=\"{:.1f}\" stroke=\"#e6e5e1\" stroke-width=\"1\"/>\n", px, Y(q), px + pw, Y(q));
      if (p == 0) o += std::format("<text x=\"{:.1f}\" y=\"{:.1f}\" font-size=\"11\" fill=\"#52514e\" text-anchor=\"end\">{:.0f}</text>\n", px - 6, Y(q) + 4, q);
    }
    for (const auto& [b, lab] : std::vector<std::pair<double, const char*>>{{100, "100 B"}, {1e3, "1 KB"}, {1e4, "10 KB"}, {1e5, "100 KB"}, {1e6, "1 MB"}}) {
      if (std::log10(b) < x0 || std::log10(b) > x1) continue;
      o += std::format("<line x1=\"{:.1f}\" y1=\"{:.1f}\" x2=\"{:.1f}\" y2=\"{:.1f}\" stroke=\"#e6e5e1\" stroke-width=\"1\"/>\n", X(b), mt, X(b), mt + ph);
      o += std::format("<text x=\"{:.1f}\" y=\"{:.1f}\" font-size=\"11\" fill=\"#52514e\" text-anchor=\"middle\">{}</text>\n", X(b), mt + ph + 16, lab);
    }
    o += std::format("<line x1=\"{:.1f}\" y1=\"{:.1f}\" x2=\"{:.1f}\" y2=\"{:.1f}\" stroke=\"#8a8984\" stroke-width=\"1\"/>\n", px, mt + ph, px + pw, mt + ph);
    o += std::format("<text x=\"{:.1f}\" y=\"{:.1f}\" font-size=\"11\" fill=\"#52514e\" text-anchor=\"middle\">bytes per {} (log scale)</text>\n", px + pw / 2,
                     mt + ph + 34, unit);
    if (p == 0) {
      o += std::format("<text transform=\"translate({:.1f},{:.1f}) rotate(-90)\" font-size=\"11\" fill=\"#52514e\" text-anchor=\"middle\">active PSNR (dB)</text>\n",
                       px - 36, mt + ph / 2);
    }
    const auto& ser = panels[p].second;
    for (std::size_t i = ser.size(); i-- > 0;) {  // the hero (first) drawn last, on top
      if (i >= 8 || ser[i].pts.empty()) continue;
      std::vector<std::pair<double, double>> shown;  // points above the scale (lossless) are not drawn
      for (const auto& pt : ser[i].pts) {
        if (pt.second <= y1) shown.push_back(pt);
      }
      std::string pl;
      for (const auto& [b, q] : shown) pl += std::format("{:.1f},{:.1f} ", X(b), Y(q));
      o += std::format("<polyline points=\"{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"{}\" stroke-linejoin=\"round\"{}/>\n", pl, colors[i], ser[i].hero ? 3 : 2,
                       ser[i].dashed ? " stroke-dasharray=\"5 3\"" : "");
      for (const auto& [b, q] : shown) {
        o += std::format("<circle cx=\"{:.1f}\" cy=\"{:.1f}\" r=\"{}\" fill=\"{}\" stroke=\"#fcfcfb\" stroke-width=\"2\"><title>{}: {:.0f} B, {:.2f} dB</title></circle>\n",
                         X(b), Y(q), ser[i].hero ? 4.5 : 4, colors[i], ser[i].label, b, q);
      }
    }
  }
  o += "</svg>\n";
  fs::create_directories(path.parent_path());
  std::ofstream(path) << o;
}

// Families drawn in the figure, in the palette's order (at most 8). The video formats are those chosen on validation.
std::vector<std::pair<std::string, std::string>> figure_families() {
  return {{"g3a", "G3a"},
          {"x264_444", "H.264 4:4:4"},
          {"x265_444", "H.265 4:4:4"},
          {"vp9a", "VP9 alpha"},
          {"aom_444", "AV1 4:4:4"},
          {"x265_444_64px", "H.265 4:4:4 at 64 px"},
          {"flipbook_bc3_packed", "flipbook BC3, packed"},
          {"flipbook_raw_packed", "flipbook raw, packed"}};
}

// Paired comparisons of the `hero` family with every other family, per run: bytes needed at stated qualities (log ratio,
// paired bootstrap over runs) and active PSNR at stated byte budgets.
void compare_families(const std::string& label, const std::vector<Point>& pts, const std::vector<std::string>& runs,
                      std::map<std::string, std::set<std::string>>& fam, const std::string& hero, const std::vector<double>& qualities,
                      const std::vector<double>& rates, std::vector<std::string>& rows_q, std::vector<std::string>& rows_r) {
  for (const auto& [meth, cfgs] : fam) {
    if (meth == hero) continue;
    for (const double Q : qualities) {
      std::vector<double> x, y;
      int floors = 0, g_missing = 0, o_missing = 0;
      for (const auto& run : runs) {
        const Curve cg = curve(pts, run, hero, fam[hero], 0), co = curve(pts, run, meth, cfgs, 0);
        bool fg = false, fo = false;
        const double bg = bytes_at(cg, Q, fg), bo = bytes_at(co, Q, fo);
        if (std::isnan(bg)) ++g_missing;
        if (std::isnan(bo)) ++o_missing;
        if (std::isnan(bg) || std::isnan(bo)) continue;
        if (fo) ++floors;
        x.push_back(bg);
        y.push_back(bo);
      }
      if (x.size() < 3) {
        rows_q.push_back(std::format("{},{},{:.0f},{},{},{},,,,,,", label, meth, Q, x.size(), g_missing, o_missing));
        continue;
      }
      const auto iv = metrics::paired_bootstrap(x, y);
      double mg = 0, mo = 0;
      for (std::size_t i = 0; i < x.size(); ++i) {
        mg += std::exp(x[i]) / static_cast<double>(x.size());
        mo += std::exp(y[i]) / static_cast<double>(x.size());
      }
      rows_q.push_back(std::format("{},{},{:.0f},{},{},{},{},{:.0f},{:.0f},{:.3f},{:.3f},{:.3f}", label, meth, Q, x.size(), g_missing, o_missing, floors, mg, mo,
                                   std::exp(iv.mean), std::exp(iv.lo), std::exp(iv.hi)));
    }
    for (const double R : rates) {
      std::vector<double> x, y;
      for (const auto& run : runs) {
        const double qg = quality_at(curve(pts, run, hero, fam[hero], 0), std::log(R));
        const double qo = quality_at(curve(pts, run, meth, cfgs, 0), std::log(R));
        if (std::isnan(qg) || std::isnan(qo)) continue;
        x.push_back(qg);
        y.push_back(qo);
      }
      if (x.size() < 3) {
        rows_r.push_back(std::format("{},{},{:.0f},{},,,,,", label, meth, R, x.size()));
        continue;
      }
      const auto iv = metrics::paired_bootstrap(x, y);
      double mg = 0, mo = 0;
      for (std::size_t i = 0; i < x.size(); ++i) {
        mg += x[i] / static_cast<double>(x.size());
        mo += y[i] / static_cast<double>(x.size());
      }
      rows_r.push_back(std::format("{},{},{:.0f},{},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}", label, meth, R, x.size(), mg, mo, iv.mean, iv.lo, iv.hi));
    }
  }
}

std::vector<std::string> run_ids_of(const std::vector<RunSpec>& runs) {
  std::vector<std::string> v;
  for (const auto& r : runs) v.push_back(r.id);
  return v;
}

// G3b's tables and figure from g3b.csv: families are the model plus residual (with the model alone as its first point),
// each video codec, and the flipbook envelopes; paired over the 12 clips.
void summary_g3b(const fs::path& dir, const fs::path& res, const std::string& figure) {
  std::vector<Point> pts;
  std::set<std::string> clips;
  std::set<std::string> seen;
  for (const auto& r : read_csv(dir / "g3b.csv")) {
    std::string method = r.at("method"), config = r.at("config");
    if (!seen.insert(r.at("clip") + "|" + method + "|" + config).second) continue;
    if (method == "model_packed") {
      method = "g3b";
      config += "_alone";
    }
    pts.push_back({r.at("clip"), method, config, std::stod(r.at("bytes")), std::stod(r.at("active_psnr")), std::stod(r.at("psnr")), std::stod(r.at("ssim"))});
    clips.insert(r.at("clip"));
  }
  if (pts.empty()) return;
  const std::vector<std::string> runs(clips.begin(), clips.end());
  const auto ms = means(pts, static_cast<int>(runs.size()));
  std::map<std::string, std::set<std::string>> fam;
  std::set<std::string> methods;
  for (const auto& m : ms) methods.insert(m.method);
  for (const auto& meth : methods) {
    if (meth == "model" || meth == "residual_only") continue;
    if (meth.rfind("flipbook", 0) == 0) {
      for (const auto& c : frontier(ms, meth)) fam[meth].insert(c);
    } else {
      for (const auto& m : ms) {
        if (m.method == meth) fam[meth].insert(m.config);
      }
    }
  }
  std::vector<std::string> rows_curve, rows_q, rows_r;
  for (const auto& m : ms) {
    rows_curve.push_back(std::format("{},{},{:.1f},{:.3f},{:.3f},{:.4f}", m.method, m.config, m.bytes, m.apsnr, m.psnr, m.ssim));
  }
  compare_families("study_a_clips", pts, runs, fam, "g3b", {24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50},
                   {10000, 30000, 100000, 200000, 300000, 500000},
                   rows_q, rows_r);
  const auto write = [&](const fs::path& p, const std::string& head, const std::vector<std::string>& rows) {
    std::ofstream o(p);
    o << head << "\n";
    for (const auto& r : rows) o << r << "\n";
  };
  write(res / "g3b_curves.csv", "method,config,bytes,active_psnr,psnr,ssim", rows_curve);
  write(res / "g3b_at_quality.csv", "clips,other,active_psnr,runs,g3b_unreached,other_unreached,other_at_floor,g3b_bytes,other_bytes,bytes_ratio,ratio_lo,ratio_hi",
        rows_q);
  write(res / "g3b_at_rate.csv", "clips,other,bytes,runs,g3b_active_psnr,other_active_psnr,diff,diff_lo,diff_hi", rows_r);
  std::vector<Series> ser;
  const std::vector<std::pair<std::string, std::string>> order = {{"g3b", "G3b"},
                                                                  {"x264_rgb", "H.264 RGB"},
                                                                  {"x265_444", "H.265 4:4:4"},
                                                                  {"vp9a", "VP9 alpha"},
                                                                  {"aom_444", "AV1 4:4:4"},
                                                                  {"svt", "SVT-AV1 4:2:0"},
                                                                  {"flipbook_bc3_packed", "flipbook BC3, packed"},
                                                                  {"flipbook_raw_packed", "flipbook raw, packed"}};
  for (const auto& [meth, lab] : order) {
    if (!fam.count(meth)) continue;
    Series x;
    x.label = lab;
    x.hero = meth == "g3b";
    x.dashed = meth.rfind("flipbook", 0) == 0;
    for (const auto& m : ms) {
      if (m.method == meth && fam[meth].count(m.config)) x.pts.emplace_back(m.bytes, m.apsnr);
    }
    std::ranges::sort(x.pts);
    ser.push_back(std::move(x));
  }
  write_svg(figure, {{"12 study A clips (64 frames, 128 x 128)", ser}}, "G3b: a frame model plus a coded residual, against video codecs and flipbooks",
            "Means over the 12 study A clips. G3b bytes include the packed frame model (grid_m 8-bit, about 87 KB). Lossless points (crf 0) are off the scale.",
            55, "clip");
}

void cmd_summary(const tools::Args& a) {
  const fs::path dir = a.str("data", g3_dir());
  const fs::path res = a.str("results", "results/experiments");
  fs::create_directories(res);
  // 1. validation: the codec's frontier (written for the test run) and the variants
  {
    std::vector<std::string> rows_front, rows_var, rows_val, rows_vid;
    for (const auto e : effects_of(a.str("effects"))) {
      const std::string en(sim::effect_name(e));
      const auto pts = load_points(dir, "val", e);
      if (pts.empty()) continue;
      std::set<std::string> run_ids;
      for (const auto& p : pts) {
        if (p.method == "g3a") run_ids.insert(p.run);
      }
      const int n = static_cast<int>(run_ids.size());
      const auto ms = means(pts, n);
      for (const auto& m : ms) {
        if (m.method == "g3a") rows_val.push_back(std::format("{},{},{:.1f},{:.3f},{:.3f},{:.4f}", en, m.config, m.bytes, m.apsnr, m.psnr, m.ssim));
      }
      {  // video formats on the validation runs they were run on (the choice of formats for the test)
        std::vector<Point> vp;
        std::set<std::string> vr;
        for (const auto& p : pts) {
          if (p.method != "g3a") {
            vp.push_back(p);
            vr.insert(p.run);
          }
        }
        for (const auto& m : means(vp, static_cast<int>(vr.size()))) {
          rows_vid.push_back(std::format("{},{},{},{},{:.1f},{:.3f},{:.3f}", en, m.method, m.config, vr.size(), m.bytes, m.apsnr, m.psnr));
        }
      }
      const auto fr = frontier(ms, "g3a");
      std::ofstream f(dir / std::format("frontier_{}.txt", en));
      for (const auto& c : fr) {
        f << c << "\n";
        const auto it = std::ranges::find_if(ms, [&](const Mean& m) { return m.method == "g3a" && m.config == c; });
        rows_front.push_back(std::format("{},{},{:.1f},{:.3f},{:.3f},{:.4f}", en, c, it->bytes, it->apsnr, it->psnr, it->ssim));
      }
      // variants against their base, paired over runs
      const auto base_of = [](const std::string& c) -> std::string {
        codec::Settings s = parse_settings(c), b = base_settings();
        if (s.kf > 0) {
          b.k = 0;
          b.q = 0.4f;
          b.kf = 1;
          b.fine_res = 128;
          b.qf = 0.07f;
        } else {
          b.k = s.k;
          b.q = s.q;
        }
        return codec::describe(b);
      };
      for (const auto& c : ladder_set("variants")) {
        const std::string name = codec::describe(c), base = base_of(name);
        std::vector<double> lb_v, lb_b, q_v, q_b;
        for (const auto& rs : split_runs(e, "val")) {
          const auto get = [&](const std::string& cfg) -> const Point* {
            for (const auto& p : pts) {
              if (p.run == rs.id && p.method == "g3a" && p.config == cfg) return &p;
            }
            return nullptr;
          };
          const Point *pv = get(name), *pb = get(base);
          if (!pv || !pb) continue;
          lb_v.push_back(std::log(pv->bytes));
          lb_b.push_back(std::log(pb->bytes));
          q_v.push_back(pv->apsnr);
          q_b.push_back(pb->apsnr);
        }
        if (lb_v.size() < 2) continue;
        const auto db = metrics::paired_bootstrap(lb_v, lb_b), dq = metrics::paired_bootstrap(q_v, q_b);
        rows_var.push_back(std::format("{},{},{},{},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}", en, name, base, lb_v.size(), std::exp(db.mean), std::exp(db.lo),
                                       std::exp(db.hi), dq.mean, dq.lo, dq.hi));
      }
    }
    std::ofstream(res / "g3_val_ladder.csv") << "effect,settings,bytes,active_psnr,psnr,ssim\n" << [&] {
      std::string t;
      for (const auto& r : rows_val) t += r + "\n";
      return t;
    }();
    std::ofstream(res / "g3_val_frontier.csv") << "effect,settings,bytes,active_psnr,psnr,ssim\n" << [&] {
      std::string t;
      for (const auto& r : rows_front) t += r + "\n";
      return t;
    }();
    std::ofstream(res / "g3_val_video.csv") << "effect,method,config,runs,bytes,active_psnr,psnr\n" << [&] {
      std::string t;
      for (const auto& r : rows_vid) t += r + "\n";
      return t;
    }();
    std::ofstream(res / "g3_val_variants.csv") << "effect,variant,base,runs,bytes_ratio,bytes_ratio_lo,bytes_ratio_hi,d_active_psnr,d_lo,d_hi\n" << [&] {
      std::string t;
      for (const auto& r : rows_var) t += r + "\n";
      return t;
    }();
  }
  // 2. test: curves (means per config), and paired comparisons at stated qualities and rates
  std::vector<std::string> rows_curve, rows_q, rows_r, rows_runs;
  std::vector<std::pair<std::string, std::vector<Series>>> fig;
  for (const auto e : effects_of(a.str("effects"))) {
    const std::string en(sim::effect_name(e));
    const auto all_pts = load_points(dir, "test", e);
    if (all_pts.empty()) continue;
    // the test runs: those the codec was run on (study B's held-out settings; the salt-2 runs if they were added)
    std::set<std::string> ids;
    for (const auto& p : all_pts) {
      if (p.method == "g3a") ids.insert(p.run);
    }
    std::vector<RunSpec> runs;
    for (const auto& r : split_runs(e, "test")) {
      if (ids.count(r.id)) runs.push_back(r);
    }
    std::vector<Point> pts;
    for (const auto& p : all_pts) {
      if (ids.count(p.run)) pts.push_back(p);
    }
    const int n = static_cast<int>(runs.size());
    const auto ms = means(pts, n);
    // families and their configs: the codec's validation frontier; every rung of each video codec; the flipbook
    // envelope of test means (generous to flipbooks)
    std::map<std::string, std::set<std::string>> fam;
    {
      std::ifstream f(dir / std::format("frontier_{}.txt", en));
      for (std::string l; std::getline(f, l);) {
        if (!l.empty()) fam["g3a"].insert(l);
      }
    }
    std::set<std::string> methods;
    for (const auto& m : ms) methods.insert(m.method);
    for (const auto& meth : methods) {
      if (meth == "g3a") continue;
      if (meth.rfind("flipbook", 0) == 0) {
        for (const auto& c : frontier(ms, meth)) fam[meth].insert(c);
      } else {
        for (const auto& m : ms) {
          if (m.method == meth) fam[meth].insert(m.config);
        }
      }
    }
    for (const auto& p : pts) {
      if (fam.count(p.method) && fam[p.method].count(p.config)) {
        rows_runs.push_back(std::format("{},{},{},{},{:.0f},{:.3f},{:.3f},{:.4f}", en, p.run, p.method, p.config, p.bytes, p.apsnr, p.psnr, p.ssim));
      }
    }
    for (const auto& m : ms) {
      if (!fam.count(m.method) || !fam[m.method].count(m.config)) continue;
      rows_curve.push_back(std::format("{},{},{},{:.1f},{:.3f},{:.3f},{:.4f}", en, m.method, m.config, m.bytes, m.apsnr, m.psnr, m.ssim));
    }
    {  // the figure's series, in a fixed order of families (the codec first)
      std::vector<Series> ser;
      for (const auto& [meth, label] : figure_families()) {
        if (!fam.count(meth)) continue;
        Series x;
        x.label = label;
        x.hero = meth == "g3a";
        x.dashed = meth.rfind("flipbook", 0) == 0;
        for (const auto& m : ms) {
          if (m.method == meth && fam[meth].count(m.config)) x.pts.emplace_back(m.bytes, m.apsnr);
        }
        std::ranges::sort(x.pts);
        ser.push_back(std::move(x));
      }
      fig.emplace_back(en, std::move(ser));
    }
    compare_families(en, pts, run_ids_of(runs), fam, "g3a", {14, 16, 18, 20, 22, 24, 26, 28, 30, 33, 36},
                     {100, 300, 1000, 3000, 10000, 30000, 100000, 300000}, rows_q, rows_r);
  }
  const auto write = [&](const fs::path& p, const std::string& head, const std::vector<std::string>& rows) {
    std::ofstream o(p);
    o << head << "\n";
    for (const auto& r : rows) o << r << "\n";
  };
  write(res / "g3_test_curves.csv", "effect,method,config,bytes,active_psnr,psnr,ssim", rows_curve);
  write(res / "g3_test_runs.csv", "effect,run,method,config,bytes,active_psnr,psnr,ssim", rows_runs);
  write(res / "g3_test_at_quality.csv",
        "effect,other,active_psnr,runs,g3a_unreached,other_unreached,other_at_floor,g3a_bytes,other_bytes,bytes_ratio,ratio_lo,ratio_hi", rows_q);
  write(res / "g3_test_at_rate.csv", "effect,other,bytes,runs,g3a_active_psnr,other_active_psnr,diff,diff_lo,diff_hi", rows_r);
  if (!fig.empty()) {
    write_svg(a.str("figure", "docs/figures/g3_rd.svg"), fig, "G3a: a run codec from the learned dynamics, against video codecs and flipbooks",
              "Test: study B's 10 held-out settings, new seeds; 240 frames at 128 x 128 (explosions 89); means over the runs. Codec points: the frontier chosen on validation.");
  }
  summary_g3b(dir, res, a.str("figure-g3b", "docs/figures/g3b_rd.svg"));
  if (fs::exists(dir / "timing.csv")) fs::copy_file(dir / "timing.csv", res / "g3_timing.csv", fs::copy_options::overwrite_existing);
  std::println("summary written to {}", res.string());
}

// --- timing -----------------------------------------------------------------------------------------------------------------

// Decode time per frame of the run codec (thread CPU time of the whole decode: integers, the effect's step and render,
// the least of `reps` repetitions) pinned to one core, and the video decoders' (ffmpeg -benchmark, decode only, pinned).
void cmd_timing(const tools::Args& a) {
  const fs::path models = a.str("models", models_dir());
  const fs::path dir = a.str("data", g3_dir());
  const fs::path work = a.str("work", work_dir());
  const int reps = a.i("reps", 5), cpu = a.i("cpu", 3);
  const auto vcfg = video_configs(a.str("codecs", ""));
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  if (sched_setaffinity(0, sizeof set, &set) != 0) std::println(stderr, "warning: could not pin to cpu {}", cpu);
  double load[3] = {0, 0, 0};
  getloadavg(load, 3);
  Appender app(dir / "timing.csv", "effect,run,method,config,bytes,frames,ms_per_frame,entropy_ms_per_frame,working_kb,resident_kb,peak_rss_kb,load,reps,cpu");
  for (const auto e : effects_of(a.str("effects"))) {
    const std::string en(sim::effect_name(e));
    const rollout::Model M = load_effect(models, e);
    const auto rs = split_runs(e, "test").front();
    const Simulated S = simulate(rs);
    std::vector<std::string> pts;
    {
      std::vector<std::string> all;
      std::ifstream f(dir / std::format("frontier_{}.txt", en));
      for (std::string l; std::getline(f, l);) {
        if (!l.empty()) all.push_back(l);
      }
      const int every = std::max(1, a.i("every", 1));  // every n-th frontier point, and the last
      for (std::size_t i = 0; i < all.size(); ++i) {
        if (i % sz(every) == 0 || i + 1 == all.size()) pts.push_back(all[i]);
      }
    }
    // resident: the effect as loaded (weights and start points in floats) plus the stream
    std::size_t model_floats = M.step_w.size() + M.render_w.size();
    for (const auto& sp : M.starts) model_floats += sp.coarse.size() + sp.fine_t.size() + sp.fine_d.size();
    for (const auto& name : pts) {
      const auto enc = codec::encode(M, S.run, parse_settings(name));
      double best = 1e30, ent = 0;
      std::size_t work_b = 0;
      for (int r = 0; r < reps; ++r) {
        const double t0 = thread_seconds();
        const auto d = codec::decode(M, enc.stream);
        const double t = thread_seconds() - t0;
        if (!d) throw std::runtime_error(d.error());
        if (t < best) {
          best = t;
          ent = d->entropy_seconds;
          work_b = d->working_bytes;
        }
      }
      getloadavg(load, 3);
      app.add({std::format("{},{},g3a,{},{},{},{:.3f},{:.3f},{:.0f},{:.0f},,{:.2f},{},{}", en, rs.id, name, enc.stream.size(), rs.frames, 1e3 * best / rs.frames,
                           1e3 * ent / rs.frames, static_cast<double>(work_b) / 1024.0,
                           static_cast<double>(model_floats * sizeof(float) + enc.stream.size()) / 1024.0, load[0], reps, cpu)});
      std::println("timing {} {}: {:.2f} ms/frame", en, name, 1e3 * best / rs.frames);
      std::fflush(stdout);
    }
    for (const auto& vc : vcfg) {
      for (const int q : vc.codec.ladder) {
        std::vector<std::uint8_t> src;
        std::span<const std::uint8_t> in = S.ref.rgba;
        if (vc.res != kSize) {
          src = downsample_clip(S.ref, vc.res);
          in = src;
        }
        const auto r = video::roundtrip(vc.codec, q, in, vc.res, rs.frames, rs.p.fps, work, en);
        if (!r.ok) throw std::runtime_error(r.error);
        const fs::path file = work / std::format("{}_{}_q{}.{}", en, vc.codec.name, q, vc.codec.ext);
        const auto [secs, rss] = video::decode_cpu_seconds(file, vc.codec.decoder, reps, cpu);
        getloadavg(load, 3);
        app.add({std::format("{},{},{},{},{},{},{:.3f},,,,{:.0f},{:.2f},{},{}", en, rs.id, vc.name(), q, r.bytes, rs.frames, 1e3 * secs / rs.frames, rss, load[0],
                             reps, cpu)});
        std::error_code ec;
        fs::remove(file, ec);
      }
    }
  }
}

void cmd_probe(const tools::Args& a) {
  const fs::path models = a.str("models", models_dir());
  const auto e = effects_of(a.str("effects", "fire")).front();
  const rollout::Model M = load_effect(models, e);
  std::println("model {}: scale {} {} {} {}  render_scale {} {}  starts {}  start_fine {}", sim::effect_name(e), M.scale[0], M.scale[1], M.scale[2],
               M.scale[3], M.render_scale[0], M.render_scale[1], M.starts.size(), M.h.start_fine);
  const auto runs = split_runs(e, "val");
  const RunSpec& rs = runs[sz(a.i("run", 0))];
  const double t0 = thread_seconds();
  const Simulated S = simulate(rs);
  std::println("simulated {} frames in {:.2f} s", rs.frames, thread_seconds() - t0);
  std::vector<codec::Settings> pts;
  if (a.has("settings")) {
    std::stringstream ss(a.str("settings"));
    for (std::string t; std::getline(ss, t, ';');) {
      if (!t.empty()) pts.push_back(parse_settings(t));
    }
  }
  else pts = {[] { auto s = base_settings(); s.k = 0; return s; }(), base_settings(), [] { auto s = base_settings(); s.k = 1; s.q = 0.05f; return s; }(),
              [] { auto s = base_settings(); s.k = 1; s.q = 0.05f; s.kf = 1; s.fine_res = 128; s.qf = 0.05f; return s; }()};
  for (const auto& s : pts) {
    const double t1 = thread_seconds();
    const auto enc = codec::encode(M, S.run, s, true);
    const double te = thread_seconds() - t1;
    const double t2 = thread_seconds();
    const auto dec = codec::decode(M, enc.stream);
    const double td = thread_seconds() - t2;
    const auto sc = metrics::score(S.ref, as_clip(enc.frames, rs.frames));
    std::println("{}: {} B (start {:.0f}+{:.0f}, coarse {:.0f}, fine {:.0f}), apsnr {:.2f} psnr {:.2f} ssim {:.4f}; encode {:.1f} ms/f, decode {:.1f} ms/f, "
                 "same {} verified {}",
                 codec::describe(s), enc.stream.size(), enc.coarse_start_bytes, enc.fine_start_bytes, enc.coarse_bytes, enc.fine_bytes, sc.active_psnr,
                 sc.psnr, sc.ssim, 1e3 * te / rs.frames, 1e3 * td / rs.frames, dec && dec->rgba == enc.frames, dec && dec->verified);
  }
}

void cmd_probe_video(const tools::Args& a) {
  const auto e = effects_of(a.str("effects", "fire")).front();
  const auto runs = split_runs(e, "val");
  const RunSpec& rs = runs[sz(a.i("run", 0))];
  const Simulated S = simulate(rs);
  const fs::path work = a.str("work", work_dir());
  const std::string only = a.str("codecs");
  for (const auto& c : video::study_codecs()) {
    if (!only.empty() && only.find(c.name) == std::string::npos) continue;
    for (const int q : c.ladder) {
      const auto t0 = std::chrono::steady_clock::now();
      const auto r = video::roundtrip(c, q, S.ref.rgba, kSize, rs.frames, 30.f, work, "probe");
      const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      if (!r.ok) {
        std::println("{} q{}: {}", c.name, q, r.error);
        continue;
      }
      const auto sc = metrics::score(S.ref, as_clip(r.rgba, rs.frames));
      std::println("{} q{}: {} B ({} file), apsnr {:.2f} psnr {:.2f} ssim {:.4f}; encode {:.2f} s cpu, decode {:.2f} s cpu, wall {:.1f} s", c.name, q, r.bytes,
                   r.file_bytes, sc.active_psnr, sc.psnr, sc.ssim, r.encode_cpu_s, r.decode_cpu_s, wall);
      std::fflush(stdout);
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const tools::Args a(argc, argv, {"quick", "keep", "no-ssim"});
    const std::string cmd = a.positional().empty() ? "" : a.positional()[0];
    if (cmd == "probe") cmd_probe(a);
    else if (cmd == "ladder") cmd_ladder(a);
    else if (cmd == "probe-video") cmd_probe_video(a);
    else if (cmd == "baselines") cmd_baselines(a);
    else if (cmd == "summary") cmd_summary(a);
    else if (cmd == "timing") cmd_timing(a);
    else if (cmd == "g3b") cmd_g3b(a);
    else {
      std::println(stderr, "usage: nvfx_g3 probe|ladder|baselines|summary|timing [options]");
      return 2;
    }
    a.warn_unused();
  } catch (const std::exception& ex) {
    std::println(stderr, "nvfx_g3: {}", ex.what());
    return 1;
  }
  return 0;
}
