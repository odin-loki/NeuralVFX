// nvfx_g3: study G3 (docs/DCM.md §8), a codec for authored runs built from the learned dynamics, measured as
// rate-distortion curves against flipbooks and video codecs on the same runs.
//
//   probe      one run, a few settings: bytes, quality and time (a sanity check)
//   ladder     the run codec over a ladder of settings, every run of a split; rows appended per run (resumable)
//   baselines  flipbooks (BC3, raw RGBA; as stored and packed by the model-file coder) and video codecs through ffmpeg,
//              every run of a split; rows appended per run (resumable)
//   summary    curves, the validation-chosen frontier, paired comparisons on test, CSVs and the figure
//   timing     decode time per frame and memory of the run codec on one pinned core
//
// Splits (docs/DCM.md §4): val = the 10 validation settings of study G with seeds 800000 + i; test = study B's 10 held-out
// settings with seeds 900000 + i and study D's 8 salt-2 tracking runs. Runs: 240 frames after a warm-up (explosions: 89
// frames after the first), the real run rendered by the simulation's own renderer at 128 x 128.
#include "args.hpp"
#include "video_pipe.hpp"

#include <neuralfx/clip.hpp>
#include <neuralfx/cm.hpp>
#include <neuralfx/codec/run_codec.hpp>
#include <neuralfx/flipbook.hpp>
#include <neuralfx/metrics.hpp>
#include <neuralfx/rollout.hpp>
#include <neuralfx/rollout_train.hpp>
#include <neuralfx/sim.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <mutex>
#include <print>
#include <random>
#include <set>
#include <sstream>
#include <thread>

namespace fs = std::filesystem;
using namespace nfx;

namespace {

constexpr int kSize = 128, kRes = 32;

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
  const fs::path models = a.str("models", "/root/nvfx-data/experiments/models/d");
  const fs::path out = a.str("out", "/root/nvfx-data/g3");
  const std::string split = a.str("split", "val"), set = a.str("set", "val");
  const int threads = a.i("threads", 1);
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
    const auto runs = split_runs(e, split);
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
        const auto sc = metrics::score(S.ref, as_clip(enc.frames, rs.frames));
        app.add({ladder_row(e, split, rs.id, *s, enc, sc, rs.frames, secs)});
      }
      std::println("ladder {} {} {}: {} points", sim::effect_name(e), split, rs.id, todo.size());
      std::fflush(stdout);
    });
  }
}

void cmd_probe(const tools::Args& a) {
  const fs::path models = a.str("models", "/root/nvfx-data/experiments/models/d");
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
  const fs::path work = a.str("work", "/tmp/claude-0/-home-user/7f3ed069-4de9-5eae-b560-587577dc6cd8/scratchpad/s4/video");
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
    const tools::Args a(argc, argv, {"quick"});
    const std::string cmd = a.positional().empty() ? "" : a.positional()[0];
    if (cmd == "probe") cmd_probe(a);
    else if (cmd == "ladder") cmd_ladder(a);
    else if (cmd == "probe-video") cmd_probe_video(a);
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
