// Study G, design G1 (docs/DCM.md): DCM-fine end to end. See fine_study.hpp.
#include "fine_study.hpp"

#include <neuralfx/clip.hpp>
#include <neuralfx/dcm/kmeans.hpp>
#include <neuralfx/dcm/search.hpp>
#include <neuralfx/image_io.hpp>
#include <neuralfx/metrics.hpp>
#include <neuralfx/rollout.hpp>
#include <neuralfx/rollout_train.hpp>

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <format>
#include <fstream>
#include <map>
#include <mutex>
#include <numeric>
#include <print>
#include <random>
#include <sstream>
#include <thread>
#include <time.h>

namespace fs = std::filesystem;

namespace nfx::fine_study {

namespace fine = dcm::fine;

namespace {

std::size_t sz(int v) { return static_cast<std::size_t>(v); }
[[maybe_unused]] float fl(int v) { return static_cast<float>(v); }
[[maybe_unused]] double dbl(float v) { return static_cast<double>(v); }

constexpr int kSize = 128;
constexpr std::size_t kMacroStatsBench = 4;
constexpr int kTrainRuns = 48, kValRuns = 16, kWindowSteps = 8;
constexpr std::uint64_t kTrainSalt = 1, kValSalt = 3;

std::string ename(sim::Effect e) { return std::string(sim::effect_name(e)); }
bool one_shot(sim::Effect e) { return !sim::effect_loops(e); }

double seconds_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

void log(const std::string& s) {
  static std::mutex m;
  const std::lock_guard lock(m);
  std::println("{}", s);
  std::fflush(stdout);
}

rollout::Model load_v1(const Ctx& c, sim::Effect e) {
  auto m = rollout::load_model(c.models / std::format("{}.nvfx", ename(e)));
  if (!m) throw std::runtime_error(m.error());
  return std::move(*m);
}

// Runs fn(i) for i in [0, n) on `threads` threads.
void parallel(int n, int threads, const std::function<void(int)>& fn) {
  std::atomic<int> next{0};
  std::exception_ptr err;
  std::mutex m;
  {
    std::vector<std::jthread> pool;
    for (int t = 0; t < std::max(1, threads); ++t) {
      pool.emplace_back([&] {
        for (int i; (i = next++) < n;) {
          try {
            fn(i);
          } catch (...) {
            const std::lock_guard lock(m);
            if (!err) err = std::current_exception();
            next = n;
          }
        }
      });
    }
  }
  if (err) std::rethrow_exception(err);
}

void write_csv(const fs::path& path, const std::string& header, const std::vector<std::string>& rows) {
  fs::create_directories(path.parent_path());
  std::ofstream o(path);
  o << header << "\n";
  for (const auto& r : rows) o << r << "\n";
}

// Rows keyed by their first `key_cells` cells replace rows with the same key; others are kept (per-effect runs merge).
[[maybe_unused]] void merge_csv(const fs::path& path, const std::string& header, const std::vector<std::string>& rows, int key_cells = 1) {
  const auto key = [&](const std::string& r) {
    std::size_t at = 0;
    for (int k = 0; k < key_cells && at != std::string::npos; ++k) {
      at = r.find(',', at == 0 && k == 0 ? 0 : at + 1);
    }
    return r.substr(0, at);
  };
  std::vector<std::string> out;
  std::map<std::string, bool> fresh;
  for (const auto& r : rows) fresh[key(r)] = true;
  if (std::ifstream in(path); in) {
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
      if (!line.empty() && !fresh.contains(key(line))) out.push_back(line);
    }
  }
  out.insert(out.end(), rows.begin(), rows.end());
  write_csv(path, header, out);
}

[[maybe_unused]] std::vector<std::vector<std::string>> read_csv(const fs::path& p) {
  std::vector<std::vector<std::string>> rows;
  std::ifstream in(p);
  std::string line;
  bool header = true;
  while (std::getline(in, line)) {
    if (header) {
      header = false;
      continue;
    }
    if (line.empty()) continue;
    std::vector<std::string> cells;
    std::size_t a = 0;
    while (true) {
      const std::size_t b = line.find(',', a);
      cells.push_back(line.substr(a, b == std::string::npos ? std::string::npos : b - a));
      if (b == std::string::npos) break;
      a = b + 1;
    }
    rows.push_back(std::move(cells));
  }
  return rows;
}

// --- runs, tables, windows -------------------------------------------------------------------------------------------

struct RunInfo {
  std::uint64_t salt = 0, index = 0, seed = 0;
  std::array<float, 3> controls{};
};

sim::Params run_params(sim::Effect e, std::uint64_t salt, std::uint64_t index) {
  rollout::SimRecipe r;
  r.effect = e;
  r.salt = salt;
  return rollout::recipe_run(r, index);  // 128 px, controls uniform in [0, 1]^3, seed of its own
}

struct Table {
  std::vector<RunInfo> runs;
  std::vector<fine::Row> rows;
};

constexpr char kTableMagic[8] = {'N', 'V', 'F', 'X', 'F', 'R', 'W', '1'};

void write_table(const fs::path& path, const Table& t) {
  static_assert(std::is_trivially_copyable_v<fine::Row> && std::is_trivially_copyable_v<RunInfo>);
  fs::create_directories(path.parent_path());
  std::ofstream o(path, std::ios::binary);
  o.write(kTableMagic, 8);
  const std::uint64_t nr = t.runs.size(), n = t.rows.size(), rs = sizeof(fine::Row);
  o.write(reinterpret_cast<const char*>(&nr), 8);
  o.write(reinterpret_cast<const char*>(&n), 8);
  o.write(reinterpret_cast<const char*>(&rs), 8);
  o.write(reinterpret_cast<const char*>(t.runs.data()), static_cast<std::streamsize>(nr * sizeof(RunInfo)));
  o.write(reinterpret_cast<const char*>(t.rows.data()), static_cast<std::streamsize>(n * sizeof(fine::Row)));
  if (!o) throw std::runtime_error("cannot write " + path.string());
}

[[maybe_unused]] Table read_table(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  char magic[8];
  in.read(magic, 8);
  if (!in || std::memcmp(magic, kTableMagic, 8) != 0) throw std::runtime_error(path.string() + " is not a row table (run record)");
  std::uint64_t nr = 0, n = 0, rs = 0;
  in.read(reinterpret_cast<char*>(&nr), 8);
  in.read(reinterpret_cast<char*>(&n), 8);
  in.read(reinterpret_cast<char*>(&rs), 8);
  if (rs != sizeof(fine::Row)) throw std::runtime_error(path.string() + ": row layout differs (record again)");
  Table t;
  t.runs.resize(nr);
  t.rows.resize(n);
  in.read(reinterpret_cast<char*>(t.runs.data()), static_cast<std::streamsize>(nr * sizeof(RunInfo)));
  in.read(reinterpret_cast<char*>(t.rows.data()), static_cast<std::streamsize>(n * sizeof(fine::Row)));
  if (!in) throw std::runtime_error(path.string() + " is truncated");
  return t;
}

// An own-rollout window: the true fine fields (and coarse state) the frame before, then kWindowSteps steps of the v1
// stepper teacher-forced along the true run, with the true fine fields of each step (targets).
struct WindowStep {
  float time = 0, since = 0;
  std::vector<float> coarse, flow, target_t, target_d;
};
struct Window {
  std::int32_t run = 0, first = 0;
  std::vector<float> fine_t, fine_d, coarse0;
  std::vector<WindowStep> steps;
};

void put_vec(std::ostream& o, const std::vector<float>& v) {
  const std::uint64_t n = v.size();
  o.write(reinterpret_cast<const char*>(&n), 8);
  o.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(n * 4));
}
std::vector<float> get_vec(std::istream& in) {
  std::uint64_t n = 0;
  in.read(reinterpret_cast<char*>(&n), 8);
  if (!in || n > (1u << 26)) throw std::runtime_error("window file: bad vector");
  std::vector<float> v(n);
  in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * 4));
  return v;
}

void write_windows(const fs::path& path, const std::vector<Window>& ws) {
  std::ofstream o(path, std::ios::binary);
  const std::uint64_t n = ws.size();
  o.write(reinterpret_cast<const char*>(&n), 8);
  for (const Window& w : ws) {
    o.write(reinterpret_cast<const char*>(&w.run), 4);
    o.write(reinterpret_cast<const char*>(&w.first), 4);
    put_vec(o, w.fine_t);
    put_vec(o, w.fine_d);
    put_vec(o, w.coarse0);
    const std::uint64_t k = w.steps.size();
    o.write(reinterpret_cast<const char*>(&k), 8);
    for (const WindowStep& s : w.steps) {
      o.write(reinterpret_cast<const char*>(&s.time), 4);
      o.write(reinterpret_cast<const char*>(&s.since), 4);
      put_vec(o, s.coarse);
      put_vec(o, s.flow);
      put_vec(o, s.target_t);
      put_vec(o, s.target_d);
    }
  }
  if (!o) throw std::runtime_error("cannot write " + path.string());
}

[[maybe_unused]] std::vector<Window> read_windows(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot read " + path.string() + " (run record)");
  std::uint64_t n = 0;
  in.read(reinterpret_cast<char*>(&n), 8);
  std::vector<Window> ws(n);
  for (Window& w : ws) {
    in.read(reinterpret_cast<char*>(&w.run), 4);
    in.read(reinterpret_cast<char*>(&w.first), 4);
    w.fine_t = get_vec(in);
    w.fine_d = get_vec(in);
    w.coarse0 = get_vec(in);
    std::uint64_t k = 0;
    in.read(reinterpret_cast<char*>(&k), 8);
    w.steps.resize(k);
    for (WindowStep& s : w.steps) {
      in.read(reinterpret_cast<char*>(&s.time), 4);
      in.read(reinterpret_cast<char*>(&s.since), 4);
      s.coarse = get_vec(in);
      s.flow = get_vec(in);
      s.target_t = get_vec(in);
      s.target_d = get_vec(in);
    }
  }
  if (!in) throw std::runtime_error(path.string() + " is truncated");
  return ws;
}

// Frames whose one-step rows are recorded, and the first frames of the own-rollout windows.
std::vector<int> sampled_frames(sim::Effect e) {
  std::vector<int> v;
  if (one_shot(e)) {
    for (int k = 1; k < 90; ++k) v.push_back(k);
  } else {
    for (int k = 60; k < 240; k += 2) v.push_back(k);
  }
  return v;
}
std::vector<int> window_starts(sim::Effect e) { return one_shot(e) ? std::vector<int>{1, 20, 40, 60} : std::vector<int>{70, 110, 150, 190}; }
int run_frames(sim::Effect e) { return one_shot(e) ? 90 : 240; }

// Seconds since the start point a frame would have in use: one-shot effects start from their first frame (the swirl
// ramps in from there), looping effects run long past the ramp.
float since_start(sim::Effect e, float time, float fps) { return one_shot(e) ? std::max(0.f, time - 1.f / fps) : time; }

fine::Spec scale_spec(const rollout::Model& m) {
  fine::Spec sp;
  for (int q = 0; q < 2; ++q) sp.s[sz(q)] = std::max(1e-6f, m.render_scale[sz(q)]);
  return sp;
}

// Sample up to `active` active and `empty` empty pixels of channel q (seeded); targets from `target`.
void sample_rows(const fine::Spec& sp, const fine::Frame& f, const rollout::State& s, int q, const std::vector<float>& target, int active,
                 int empty, std::mt19937_64& rng, int run, int frame, std::vector<fine::Row>& out) {
  std::vector<int> act, emp;
  const float thr = 1e-3f * sp.s[sz(q)];
  for (int i = 0; i < f.S * f.S; ++i) {
    if (fine::skip_pixel(sp, f, sz(i), q)) continue;
    (target[sz(i)] > thr || f.L[sz(q)][sz(i)] > thr ? act : emp).push_back(i);
  }
  const auto take = [&](std::vector<int>& v, int n, bool is_empty) {
    n = std::min(n, static_cast<int>(v.size()));
    for (int k = 0; k < n; ++k) {
      const int j = k + static_cast<int>(rng() % static_cast<std::uint64_t>(static_cast<int>(v.size()) - k));
      std::swap(v[sz(k)], v[sz(j)]);
      const int i = v[sz(k)];
      fine::Row r = fine::row_of(f, s, i % f.S, i / f.S, q);
      r.run = run;
      r.frame = static_cast<std::int16_t>(frame);
      r.empty = is_empty ? 1 : 0;
      r.target = target[sz(i)];
      out.push_back(r);
    }
  };
  take(act, active, false);
  take(emp, empty, true);
}

struct RunRecord {
  std::vector<fine::Row> rows;
  std::vector<Window> windows;
};

// One run: the simulation at 128 px; every frame the v1 stepper takes the true coarse state (memory channels carried
// from its own previous step), and the detail step's experts are computed from the true fine fields of the frame before.
RunRecord record_run(const rollout::Model& M, sim::Effect e, std::uint64_t salt, std::uint64_t index, int run_id, bool windows, int active,
                     int empty) {
  const sim::Params p = run_params(e, salt, index);
  const std::vector<float> controls{p.intensity, p.wind, p.turbulence};
  const int R = M.h.res, C = M.h.channels(), F = run_frames(e);
  const std::size_t N = sz(R) * sz(R), S2 = sz(kSize) * sz(kSize);
  const fine::Spec sp = scale_spec(M);
  const auto frames = sampled_frames(e);
  const auto wstarts = windows ? window_starts(e) : std::vector<int>{};
  RunRecord out;
  sim::Fluid fluid(p);
  std::vector<float> in(N * sz(C), 0.f), outc(N * sz(C)), flow(N * 2), pressure(N, 0.f), cond(sz(M.h.cond())), noise(N * rollout::kNoise);
  std::vector<float> prev_t(S2, 0.f), prev_d(S2, 0.f), prev_coarse(N * rollout::kPhys, 0.f), cur_coarse(N * rollout::kPhys);
  fine::Frame fr;
  std::mt19937_64 rng(p.seed * 1000003ULL + 17);
  Window* open = nullptr;
  for (int k = 0; k < F; ++k) {
    fluid.step_frame();
    const sim::State st = fluid.state();
    rollout::coarse_from_sim(st, R, p.fps, cur_coarse);
    // the stepper from the true coarse state of the frame before (time k / fps)
    const float time = static_cast<float>(k) / p.fps;
    for (std::size_t i = 0; i < N; ++i) {
      for (int c = 0; c < rollout::kPhys; ++c) in[i * sz(C) + sz(c)] = prev_coarse[i * rollout::kPhys + sz(c)];
    }
    rollout::condition(M, controls, time, cond);
    rollout::coarse_noise(M, p.seed, time, noise);
    rollout::coarse_step(M, in, noise, cond, pressure, outc, flow);
    for (std::size_t i = 0; i < N; ++i) {
      for (int c = rollout::kPhys; c < C; ++c) in[i * sz(C) + sz(c)] = outc[i * sz(C) + sz(c)];
    }
    if (std::ranges::find(wstarts, k) != wstarts.end()) {
      Window w;
      w.run = run_id;
      w.first = k;
      w.fine_t = prev_t;
      w.fine_d = prev_d;
      w.coarse0 = prev_coarse;
      out.windows.push_back(std::move(w));
      open = &out.windows.back();
    }
    if (open && static_cast<int>(open->steps.size()) < kWindowSteps) {
      WindowStep ws;
      ws.time = time;
      ws.since = since_start(e, time, p.fps);
      ws.coarse = outc;
      ws.flow = flow;
      ws.target_t = st.temp;
      ws.target_d = st.soot;
      open->steps.push_back(std::move(ws));
      if (static_cast<int>(open->steps.size()) == kWindowSteps) open = nullptr;
    }
    if (std::ranges::binary_search(frames, k)) {
      rollout::State s;
      s.res = R;
      s.size = kSize;
      s.time = time;
      s.since_start = since_start(e, time, p.fps);
      s.coarse = outc;
      s.flow = flow;
      s.fine_t = prev_t;
      s.fine_d = prev_d;
      fine::compute_frame(M, s, p.seed, controls, fr);
      sample_rows(sp, fr, s, 0, st.temp, active, empty, rng, run_id, k, out.rows);
      sample_rows(sp, fr, s, 1, st.soot, active, empty, rng, run_id, k, out.rows);
    }
    prev_t = st.temp;
    prev_d = st.soot;
    prev_coarse = cur_coarse;
  }
  return out;
}

fs::path table_path(const Ctx& c, sim::Effect e, std::string_view split) { return c.data / std::format("{}_{}.rows", ename(e), split); }

// A table without the rows the mixer never predicts (fine::skip_row: every value invisible).
[[maybe_unused]] Table read_rows(const Ctx& c, sim::Effect e, std::string_view split, const rollout::Model& M) {
  Table t = read_table(table_path(c, e, split));
  const fine::Spec sp = scale_spec(M);
  std::erase_if(t.rows, [&](const fine::Row& r) { return fine::skip_row(sp, r); });
  return t;
}
fs::path windows_path(const Ctx& c, sim::Effect e) { return c.data / std::format("{}_train.windows", ename(e)); }

}  // namespace

// --- record ----------------------------------------------------------------------------------------------------------

void record(const Ctx& c, sim::Effect e) {
  const rollout::Model M = load_v1(c, e);
  const int n_train = c.quick ? 6 : kTrainRuns, n_val = c.quick ? 3 : kValRuns;
  const int active = 32, empty = 4;  // per frame and channel: 64 active and 8 empty pixel rows per frame
  for (const bool val : {false, true}) {
    const int n = val ? n_val : n_train;
    const std::uint64_t salt = val ? kValSalt : kTrainSalt;
    std::vector<RunRecord> recs(sz(n));
    const auto t0 = std::chrono::steady_clock::now();
    std::atomic<int> done{0};
    parallel(n, c.threads, [&](int i) {
      recs[sz(i)] = record_run(M, e, salt, static_cast<std::uint64_t>(i), i, !val, active, empty);
      log(std::format("record {} {} run {} ({}/{}): {} rows, {:.0f} s", ename(e), val ? "validation" : "training", i, ++done, n,
                      recs[sz(i)].rows.size(), seconds_since(t0)));
    });
    Table t;
    std::vector<Window> ws;
    for (int i = 0; i < n; ++i) {
      const sim::Params p = run_params(e, salt, static_cast<std::uint64_t>(i));
      t.runs.push_back({salt, static_cast<std::uint64_t>(i), p.seed, {p.intensity, p.wind, p.turbulence}});
      auto& r = recs[sz(i)];
      t.rows.insert(t.rows.end(), r.rows.begin(), r.rows.end());
      for (auto& w : r.windows) ws.push_back(std::move(w));
      r = {};
    }
    write_table(table_path(c, e, val ? "val" : "train"), t);
    if (!val) write_windows(windows_path(c, e), ws);
    log(std::format("record {} {}: {} runs, {} rows, {} windows, {:.0f} s", ename(e), val ? "validation" : "training", n, t.rows.size(), ws.size(),
                    seconds_since(t0)));
  }
}

// --- rows prepared for the mixer ---------------------------------------------------------------------------------

namespace {

struct Prepared {
  std::vector<std::array<double, fine::kExperts>> x;
  std::vector<std::array<int, fine::kContexts>> ctx;
  std::vector<std::array<double, fine::kScaleFeatures>> z;
  std::vector<double> y;
};

Prepared prepare(const fine::Spec& sp, const Table& t) {
  Prepared p;
  const std::size_t n = t.rows.size();
  p.x.resize(n);
  p.ctx.resize(n);
  p.z.resize(n);
  p.y.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    const fine::Row& r = t.rows[i];
    const auto& run = t.runs.at(sz(r.run));
    p.ctx[i] = fine::contexts_of(sp, r, run.controls, kSize);
    fine::normalise(sp, r, p.x[i], p.z[i]);
    p.y[i] = fine::normalise_value(sp, r.channel, dbl(r.target), dbl(r.e[fine::kCoarse]));
  }
  return p;
}

fine::Domain parse_domain(const std::string& s) {
  if (s == "linear") return fine::Domain::linear;
  if (s == "log") return fine::Domain::log;
  throw std::invalid_argument("--domain linear|log");
}
fine::Family parse_family(const std::string& s) {
  if (s == "none") return fine::Family::none;
  if (s == "hand") return fine::Family::hand;
  if (s == "hand+macro" || s == "hand_macro") return fine::Family::hand_macro;
  throw std::invalid_argument("--family none|hand|hand+macro");
}

fs::path spec_path(const Ctx& c, sim::Effect e, fine::Domain d) { return c.data / std::format("{}_{}_spec.mixer", ename(e), fine::domain_name(d)); }

// The spec of an effect and domain: fitted once on the training rows, kept (inside a v1 mixer file).
fine::Spec get_spec(const Ctx& c, sim::Effect e, fine::Domain d, const rollout::Model& M, const Table& train) {
  const fs::path p = spec_path(c, e, d);
  if (fs::exists(p)) return fine::load_mixer(p).spec;
  const fine::Spec sp = fine::fit_spec(M, train.rows, d, kSize, 0);
  fine::save_mixer(p, fine::v1_mixer(sp));
  return sp;
}

// Nested search sites: k-means (K = 5) of the training runs' controls.
std::vector<int> run_sites(const Table& t, int K = 5) {
  std::vector<double> x;
  for (const auto& r : t.runs) {
    for (const float v : r.controls) x.push_back(dbl(v));
  }
  dcm::KMeansOptions o;
  o.k = std::min<int>(K, static_cast<int>(t.runs.size()));
  o.restarts = 8;
  o.seed = 0;
  return dcm::fit_kmeans(x, 3, o).labels;
}

struct Problem {
  dcm::SearchProblem p;
  std::vector<int> fam;           // problem context column -> fine::Context
  std::vector<std::size_t> rows;  // problem row -> table row
};

Problem make_problem(const fine::Spec& sp, const Prepared& pr, const Table& t, const std::vector<int>& fam, const std::vector<int>& site_of_run,
                     int max_rows) {
  Problem P;
  P.fam = fam;
  dcm::SearchProblem& p = P.p;
  const std::size_t n = t.rows.size();
  const std::size_t stride = max_rows > 0 ? std::max<std::size_t>(1, (n + sz(max_rows) - 1) / sz(max_rows)) : 1;
  for (std::size_t i = 0; i < n; i += stride) P.rows.push_back(i);
  for (int e = 0; e < fine::kExperts; ++e) p.input_names.emplace_back(fine::expert_name(e));
  for (const auto& g : fine::expert_groups()) p.groups.push_back({std::string(g.name), g.experts});
  for (const int k : fam) {
    p.context_names.emplace_back(fine::context_name(k));
    p.context_sizes.push_back(fine::context_size(sp, k));
  }
  int sites = 0;
  for (const int s : site_of_run) sites = std::max(sites, s + 1);
  for (int s = 0; s < sites; ++s) {
    p.site_names.push_back(std::format("bin{}", s));
    p.holdout.push_back(true);
  }
  std::vector<double> widths;
  for (const std::size_t i : P.rows) {
    const fine::Row& r = t.rows[i];
    p.x.emplace_back(pr.x[i].begin(), pr.x[i].end());
    std::vector<int> cx;
    for (const int k : fam) cx.push_back(pr.ctx[i][sz(k)]);
    p.contexts.push_back(std::move(cx));
    p.z.emplace_back(pr.z[i].begin(), pr.z[i].end());
    p.y.push_back(pr.y[i]);
    p.site.push_back(site_of_run.at(sz(r.run)));
    p.fold.push_back(-1);
    widths.push_back(fine::bin_width(sp, r.channel, dbl(r.target), dbl(r.e[fine::kCoarse])));
  }
  // laplace_bits codes every row in one bin width: the median of the rows' own widths (reported bits use each row's)
  std::nth_element(widths.begin(), widths.begin() + static_cast<std::ptrdiff_t>(widths.size() / 2), widths.end());
  p.delta = widths[widths.size() / 2];
  p.value_rule = fine::kLock;
  return P;
}

// Problem rows site by site, each site's in row order: the order the searches train in.
std::vector<std::size_t> site_order(const dcm::SearchProblem& p) {
  std::vector<std::size_t> rows(p.y.size());
  std::iota(rows.begin(), rows.end(), std::size_t{0});
  std::ranges::stable_sort(rows, {}, [&](std::size_t i) { return p.site[i]; });
  return rows;
}

fine::Config to_config(const Problem& P, const dcm::SearchSpace& space, const dcm::SearchConfig& sc) {
  fine::Config c;
  for (const int g : sc.groups) {
    for (const int col : P.p.groups[sz(g)].columns) c.experts.push_back(col);
  }
  std::ranges::sort(c.experts);
  c.experts.erase(std::unique(c.experts.begin(), c.experts.end()), c.experts.end());
  for (const int j : sc.mixer_contexts) c.mixer_contexts.push_back(P.fam[sz(j)]);
  c.spec = dcm::value_spec(P.p, space, sc);
  c.avm_context = sc.apm_context >= 0 && c.spec.avm_weight > 0.0 ? P.fam[sz(sc.apm_context)] : -1;
  c.scale_context = sc.scale_context >= 0 ? P.fam[sz(sc.scale_context)] : -1;
  c.epochs = space.epochs[sz(sc.epochs)];
  return c;
}

fine::Mixer train_mixer(const fine::Spec& sp, const Problem& P, const dcm::SearchSpace& space, const dcm::SearchConfig& sc, std::uint64_t seed,
                        dcm::ValueNet* keep = nullptr) {
  const auto rows = site_order(P.p);
  dcm::ValueNet net = dcm::train_value_config(P.p, space, sc, rows, seed);
  if (keep) *keep = net;
  return fine::make_mixer(sp, to_config(P, space, sc), net);
}

// Per row of a table: bits (value domain, the row's own bin) and squared error in units of s_q^2.
struct RowScores {
  std::vector<double> bits, se;
};

RowScores score_rows(const fine::Mixer& m, const Prepared& pr, const Table& t) {
  RowScores s;
  s.bits.resize(t.rows.size());
  s.se.resize(t.rows.size());
  for (std::size_t i = 0; i < t.rows.size(); ++i) {
    const fine::Row& r = t.rows[i];
    const auto p = m.predict(pr.x[i], pr.ctx[i], pr.z[i]);
    const double c = dbl(r.e[fine::kCoarse]);
    s.bits[i] = fine::value_bits(m.spec, r.channel, dbl(r.target), p.mu, p.b, c);
    const double v = std::max(0.0, fine::value_of(m.spec, r.channel, p.mu, c));
    s.se[i] = std::pow((v - dbl(r.target)) / dbl(m.spec.s[sz(r.channel)]), 2.0);
  }
  return s;
}

// The v1 lock as a predictor: mu = L, a Laplace scale per channel and lvl bin fitted on the training rows.
struct V1Predictor {
  std::array<std::array<double, 6>, 2> b{};
};
V1Predictor fit_v1(const fine::Spec& sp, const Table& train, const Prepared& pr) {
  V1Predictor v;
  std::array<std::array<double, 6>, 2> sum{}, cnt{};
  for (std::size_t i = 0; i < train.rows.size(); ++i) {
    const fine::Row& r = train.rows[i];
    const auto l = sz(pr.ctx[i][fine::kLvl]);
    sum[sz(r.channel)][l] += std::abs(dbl(r.target) - dbl(r.e[fine::kLock]));
    cnt[sz(r.channel)][l] += 1.0;
  }
  for (int q = 0; q < 2; ++q) {
    for (int l = 0; l < 6; ++l) v.b[sz(q)][sz(l)] = std::max(1e-6 * dbl(sp.s[sz(q)]), cnt[sz(q)][sz(l)] > 0 ? sum[sz(q)][sz(l)] / cnt[sz(q)][sz(l)] : 0.0);
  }
  return v;
}
RowScores score_v1(const fine::Spec& sp, const V1Predictor& v, const Prepared& pr, const Table& t) {
  RowScores s;
  for (std::size_t i = 0; i < t.rows.size(); ++i) {
    const fine::Row& r = t.rows[i];
    const double L = dbl(r.e[fine::kLock]), b = v.b[sz(r.channel)][sz(pr.ctx[i][fine::kLvl])];
    s.bits.push_back(dcm::laplace_bits(dbl(r.target), L, b, dbl(sp.s[sz(r.channel)]) / 256.0));
    s.se.push_back(std::pow((L - dbl(r.target)) / dbl(sp.s[sz(r.channel)]), 2.0));
  }
  return s;
}

// Means per run over active rows (and over every row).
struct RunMeans {
  std::vector<double> active, all, se;
  std::vector<int> n;
};
RunMeans per_run(const Table& t, const RowScores& s) {
  const std::size_t R = t.runs.size();
  RunMeans m;
  std::vector<double> a(R, 0.0), al(R, 0.0), se(R, 0.0), na(R, 0.0), nl(R, 0.0);
  for (std::size_t i = 0; i < t.rows.size(); ++i) {
    const auto r = sz(t.rows[i].run);
    if (std::isnan(s.bits[i])) continue;
    al[r] += s.bits[i];
    nl[r] += 1.0;
    if (!t.rows[i].empty) {
      a[r] += s.bits[i];
      se[r] += s.se.empty() ? 0.0 : s.se[i];
      na[r] += 1.0;
    }
  }
  for (std::size_t r = 0; r < R; ++r) {
    m.active.push_back(na[r] > 0 ? a[r] / na[r] : std::nan(""));
    m.all.push_back(nl[r] > 0 ? al[r] / nl[r] : std::nan(""));
    m.se.push_back(na[r] > 0 ? se[r] / na[r] : std::nan(""));
    m.n.push_back(static_cast<int>(na[r]));
  }
  return m;
}

double mean_of(const std::vector<double>& v) {
  double s = 0;
  int n = 0;
  for (const double x : v) {
    if (std::isfinite(x)) {
      s += x;
      ++n;
    }
  }
  return n ? s / n : std::nan("");
}

std::string iv(const metrics::Interval& v, int prec = 3) {
  return std::format("{:+.{}f} [{:+.{}f}, {:+.{}f}]{}", v.mean, prec, v.lo, prec, v.hi, prec, v.covers_zero() ? " (tie)" : "");
}

// The search's cost model from bench-experts (results/experiments/g_fine_cost.csv): components per group and context.
dcm::SearchCost cost_model(const Ctx& c, sim::Effect e, const Problem& P, bool& found) {
  dcm::SearchCost cost;
  found = false;
  const fs::path path = c.results / "g_fine_cost.csv";
  if (!fs::exists(path)) return cost;
  std::map<std::string, double> ms;
  for (const auto& r : read_csv(path)) {
    if (r.size() >= 4 && r[0] == ename(e)) ms[r[1]] = std::stod(r[3]);
  }
  if (!ms.contains("base")) return cost;
  found = true;
  cost.base_ms = ms["base"];
  for (const auto& g : fine::expert_groups()) {
    cost.components.push_back(std::format("group_{}", g.name));
    cost.ms.push_back(ms.contains(std::format("group_{}", g.name)) ? ms[std::format("group_{}", g.name)] : 0.0);
    cost.group_components.push_back({static_cast<int>(cost.components.size()) - 1});
  }
  for (const int k : P.fam) {
    const std::string name = std::format("context_{}", fine::context_name(k));
    cost.components.push_back(name);
    cost.ms.push_back(ms.contains(name) ? ms[name] : 0.0);
    cost.context_components.push_back({static_cast<int>(cost.components.size()) - 1});
  }
  cost.budget_ms = 1.0;
  return cost;
}

}  // namespace

// --- experts: the spec and quick checks ------------------------------------------------------------------------------

void experts(const Ctx& c, sim::Effect e) {
  const rollout::Model M = load_v1(c, e);
  const Table train = read_rows(c, e, "train", M), val = read_rows(c, e, "val", M);
  std::vector<std::string> rows;
  for (const auto d : {fine::Domain::linear, fine::Domain::log}) {
    const fine::Spec sp = get_spec(c, e, d, M, train);
    if (d != fine::Domain::linear) continue;
    log(std::format("experts {}: s = {:.4g} / {:.4g}, lvl edges heat {:.3g} {:.3g} {:.3g} {:.3g}, ratio edges {:.3g} {:.3g} {:.3g}, speed {:.3g} {:.3g}",
                    ename(e), sp.s[0], sp.s[1], sp.lvl[0][0], sp.lvl[0][1], sp.lvl[0][2], sp.lvl[0][3], sp.ratio[0][0], sp.ratio[0][1], sp.ratio[0][2],
                    sp.speed[0], sp.speed[1]));
    const Prepared ptr = prepare(sp, train), pv = prepare(sp, val);
    // each value-like expert alone as the predictor, a Laplace scale per channel and lvl bin (as the v1 baseline)
    for (const int ex : {fine::kAdv, fine::kAdvSl, fine::kLock, fine::kLockScaled, fine::kCoarse, fine::kPrev}) {
      std::array<std::array<double, 6>, 2> sum{}, cnt{};
      for (std::size_t i = 0; i < train.rows.size(); ++i) {
        const auto& r = train.rows[i];
        sum[sz(r.channel)][sz(ptr.ctx[i][fine::kLvl])] += std::abs(dbl(r.target) - dbl(r.e[sz(ex)]));
        cnt[sz(r.channel)][sz(ptr.ctx[i][fine::kLvl])] += 1.0;
      }
      RowScores s;
      for (std::size_t i = 0; i < val.rows.size(); ++i) {
        const auto& r = val.rows[i];
        const auto q = sz(r.channel), l = sz(pv.ctx[i][fine::kLvl]);
        const double b = std::max(1e-6 * dbl(sp.s[q]), cnt[q][l] > 0 ? sum[q][l] / cnt[q][l] : 1.0);
        s.bits.push_back(dcm::laplace_bits(dbl(r.target), dbl(r.e[sz(ex)]), b, dbl(sp.s[q]) / 256.0));
        s.se.push_back(std::pow((dbl(r.e[sz(ex)]) - dbl(r.target)) / dbl(sp.s[q]), 2.0));
      }
      const RunMeans m = per_run(val, s);
      rows.push_back(std::format("{},expert_alone,{},{:.4f},{:.5f}", ename(e), fine::expert_name(ex), mean_of(m.active), mean_of(m.se)));
    }
    // how much the regional clusters say beyond the hand-made contexts (normalised mutual information)
    for (const int mk : {fine::kMacro4, fine::kMacro8}) {
      for (const int hk : {fine::kLvl, fine::kRatio, fine::kFlow, fine::kHeight, fine::kCtrl}) {
        std::vector<int> a, b;
        for (std::size_t i = 0; i < train.rows.size(); i += 3) {
          a.push_back(ptr.ctx[i][sz(mk)]);
          b.push_back(ptr.ctx[i][sz(hk)]);
        }
        rows.push_back(std::format("{},nmi,{}~{},{:.4f},", ename(e), fine::context_name(mk), fine::context_name(hk), dcm::normalized_mutual_info(a, b)));
      }
    }
    std::array<int, 8> counts{};
    for (std::size_t i = 0; i < train.rows.size(); ++i) ++counts[sz(ptr.ctx[i][fine::kMacro8])];
    std::string cs;
    for (const int n : counts) cs += std::format(" {}", n);
    log(std::format("experts {}: macro8 cluster sizes{}", ename(e), cs));
  }
  merge_csv(c.results / "g_fine_experts.csv", "effect,kind,name,value,sq_err", rows);
  for (const auto& r : rows) log(r);
}

// --- pilot -----------------------------------------------------------------------------------------------------------

void pilot(const Ctx& c, sim::Effect e) {
  const rollout::Model M = load_v1(c, e);
  const Table train = read_rows(c, e, "train", M), val = read_rows(c, e, "val", M);
  const auto sites = run_sites(train);
  std::vector<std::string> rows;
  std::map<std::string, RunMeans> got;
  for (const auto d : {fine::Domain::linear, fine::Domain::log}) {
    const auto t0 = std::chrono::steady_clock::now();
    const fine::Spec sp = get_spec(c, e, d, M, train);
    const Prepared ptr = prepare(sp, train), pv = prepare(sp, val);
    const Problem P = make_problem(sp, ptr, train, fine::family_contexts(fine::Family::hand_macro, one_shot(e)), sites, 0);
    dcm::SearchSpace space;
    const dcm::SearchConfig def = dcm::default_config(P.p, space, dcm::Objective::laplace_bits);
    const fine::Mixer mix = train_mixer(sp, P, space, def, c.seed);
    const RunMeans m = per_run(val, score_rows(mix, pv, val));
    got[std::format("default_{}", fine::domain_name(d))] = m;
    log(std::format("pilot {} {}: default mixer {} trained in {:.0f} s ({}), validation bits per active pixel {:.4f}", ename(e), fine::domain_name(d),
                    mix.version().substr(0, 12), seconds_since(t0), fine::describe(mix.config), mean_of(m.active)));
    if (d == fine::Domain::linear) {
      const V1Predictor v1 = fit_v1(sp, train, ptr);
      got["v1_lock"] = per_run(val, score_v1(sp, v1, pv, val));
    }
  }
  for (const auto& [name, m] : got) {
    for (std::size_t r = 0; r < m.active.size(); ++r) {
      rows.push_back(std::format("{},{},{},{:.5f},{:.5f},{:.6f},{}", ename(e), name, r, m.active[r], m.all[r], m.se[r], m.n[r]));
    }
  }
  merge_csv(c.results / "g_fine_pilot.csv", "effect,method,run,bits_active,bits_all,sq_err_active,rows_active", rows);
  const auto& v1 = got["v1_lock"];
  for (const std::string name : {"default_linear", "default_log"}) {
    const auto& m = got[name];
    log(std::format("pilot {}: {} - v1_lock, bits per active pixel, paired over {} validation runs: {}; squared error: {}", ename(e), name, m.active.size(),
                    iv(metrics::paired_bootstrap(m.active, v1.active)), iv(metrics::paired_bootstrap(m.se, v1.se), 5)));
  }
  log(std::format("pilot {}: default_log - default_linear: {}", ename(e), iv(metrics::paired_bootstrap(got["default_log"].active, got["default_linear"].active))));
}

// --- search ----------------------------------------------------------------------------------------------------------

namespace {

fs::path top_path(const Ctx& c, sim::Effect e, const std::string& family, const std::string& domain, std::uint64_t seed) {
  std::string f = family;
  std::ranges::replace(f, '+', '_');
  std::ranges::replace(f, '/', '_');
  return c.data / std::format("{}_{}_{}_s{}.top", ename(e), f, domain, seed);
}

std::string encode(const dcm::SearchConfig& sc) {
  std::string s = "g";
  for (const int g : sc.groups) s += std::format(" {}", g);
  s += " m";
  for (const int j : sc.mixer_contexts) s += std::format(" {}", j);
  return s + std::format(" a {} {} {} {} {} {} {}", sc.apm_context, sc.lr1, sc.lr2, sc.epochs, sc.loss, sc.avm_weight, sc.scale_context);
}

dcm::SearchConfig decode(const std::string& line) {
  dcm::SearchConfig sc;
  std::istringstream in(line);
  std::string tok;
  in >> tok;
  if (tok != "g") throw std::runtime_error("bad config line: " + line);
  while (in >> tok && tok != "m") sc.groups.push_back(std::stoi(tok));
  while (in >> tok && tok != "a") sc.mixer_contexts.push_back(std::stoi(tok));
  in >> sc.apm_context >> sc.lr1 >> sc.lr2 >> sc.epochs >> sc.loss >> sc.avm_weight >> sc.scale_context;
  if (!in) throw std::runtime_error("bad config line: " + line);
  return sc;
}

}  // namespace

void search(const Ctx& c, sim::Effect e) {
  const rollout::Model M = load_v1(c, e);
  const Table train = read_rows(c, e, "train", M), val = read_rows(c, e, "val", M);
  const fine::Domain d = parse_domain(c.domain);
  const fine::Family fam = parse_family(c.family);
  const fine::Spec sp = get_spec(c, e, d, M, train);
  const auto sites = run_sites(train);
  const Prepared ptr = prepare(sp, train), pv = prepare(sp, val);
  const Problem P = make_problem(sp, ptr, train, fine::family_contexts(fam, one_shot(e)), sites, c.max_rows);
  dcm::SearchOptions o;
  o.configs = c.quick ? 12 : c.configs;
  o.refine_rounds = c.quick ? 1 : c.refine;
  o.top = 10;
  o.seed = c.seed;
  o.threads = c.threads;
  o.objective = dcm::Objective::laplace_bits;
  o.nested = true;
  o.global = true;
  o.rfonly = false;
  if (fam == fine::Family::none) o.space.max_mixer_contexts = 0;
  bool costed = false;
  o.cost = cost_model(c, e, P, costed);
  if (!costed) log("search: no cost model for this effect yet (run bench-experts): searching without a budget");
  const bool free = c.budget_ms <= 0.0;
  o.cost.budget_ms = free ? std::numeric_limits<double>::infinity() : c.budget_ms;
  const std::string label = (fam == fine::Family::hand_macro ? std::string("hand+macro") : c.family) + (free ? "/free" : "");
  const auto t0 = std::chrono::steady_clock::now();
  const dcm::SearchResult r = dcm::run_search(P.p, o);
  const double secs = seconds_since(t0);
  const std::string key = std::format("{},{},{},{}", ename(e), label, c.domain, c.seed);
  // nested held-out bits, per training run (value domain, each row's own bin)
  RowScores ns;
  ns.bits.assign(train.rows.size(), std::nan(""));
  ns.se.assign(train.rows.size(), std::nan(""));
  for (std::size_t k = 0; k < P.rows.size(); ++k) {
    const std::size_t i = P.rows[k];
    const fine::Row& row = train.rows[i];
    if (std::isnan(r.nested[k])) continue;
    ns.bits[i] = fine::value_bits(sp, row.channel, dbl(row.target), r.nested[k], r.nested_b[k], dbl(row.e[fine::kCoarse]));
    const double v = std::max(0.0, fine::value_of(sp, row.channel, r.nested[k], dbl(row.e[fine::kCoarse])));
    ns.se[i] = std::pow((v - dbl(row.target)) / dbl(sp.s[sz(row.channel)]), 2.0);
  }
  const RunMeans nm = per_run(train, ns);
  std::vector<std::string> run_rows, rows, val_rows;
  for (std::size_t k = 0; k < nm.active.size(); ++k) {
    run_rows.push_back(std::format("{},{},{},{:.5f},{:.5f},{:.6f}", key, k, sites[k], nm.active[k], nm.all[k], nm.se[k]));
  }
  for (const auto& ch : r.nested_choices) {
    const auto cfg = to_config(P, o.space, ch.config);
    rows.push_back(std::format("{},nested,{},{:.4f},{:.3f},\"{}\"", key, ch.site, -ch.inner_auc, dcm::config_cost(o.cost, ch.config), fine::describe(cfg)));
  }
  // the global top 10: each trained on every training row and scored on the validation runs
  std::ofstream top(top_path(c, e, free ? c.family + "_free" : c.family, c.domain, c.seed));
  std::vector<double> val_means;
  for (std::size_t k = 0; k < r.global_top.size(); ++k) {
    const auto& g = r.global_top[k];
    top << encode(g.config) << "\n";
    const fine::Mixer mix = train_mixer(sp, P, o.space, g.config, c.seed);
    const RunMeans vm = per_run(val, score_rows(mix, pv, val));
    val_means.push_back(mean_of(vm.active));
    rows.push_back(std::format("{},global,{},{:.4f},{:.3f},\"{}\"", key, k, -g.score, dcm::config_cost(o.cost, g.config), fine::describe(mix.config)));
    for (std::size_t q = 0; q < vm.active.size(); ++q) val_rows.push_back(std::format("{},{},{},{:.5f},{:.6f}", key, k, q, vm.active[q], vm.se[q]));
    log(std::format("search {}: global #{} {} (search bits {:.4f}, cost {:.3f} ms), validation bits per active pixel {:.4f}, version {}", key, k,
                    fine::describe(mix.config), -g.score, dcm::config_cost(o.cost, g.config), val_means.back(), mix.version().substr(0, 16)));
  }
  merge_csv(c.results / "g_fine_search.csv", "effect,family,domain,seed,kind,index,bits,cost_ms,config", rows, 4);
  merge_csv(c.results / "g_fine_search_runs.csv", "effect,family,domain,seed,run,site,nested_bits_active,nested_bits_all,nested_sq_err", run_rows, 4);
  merge_csv(c.results / "g_fine_topval.csv", "effect,family,domain,seed,rank,run,bits_active,sq_err", val_rows, 4);
  log(std::format("search {}: {} candidates, {} trainings in {:.0f} s; nested held-out bits per active pixel {:.4f} over {} training runs; rows {}",
                  key, r.candidates, r.trainings, secs, mean_of(nm.active), nm.active.size(), P.rows.size()));
}
// --- generation: statistics of runs against real runs -----------------------------------------------------------

namespace {

using Setting = std::array<float, 3>;

bool off_grid(const Setting& s) {
  const auto off = [](float x, std::initializer_list<float> g) { return std::ranges::all_of(g, [x](float y) { return std::abs(x - y) >= 0.05f; }); };
  return off(s[0], {0.f, 0.5f, 1.f}) && off(s[1], {0.f, 0.25f, 0.5f, 0.75f, 1.f}) && off(s[2], {0.f, 0.5f, 1.f});
}

// Study B's ten held-out settings (as nvfx_experiment's b_test_settings): the test.
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

// Ten validation settings, drawn the same way from another generator and at least 0.05 from every test setting.
std::vector<Setting> validation_settings() {
  const auto test = test_settings();
  std::mt19937_64 rng(2027);
  std::uniform_real_distribution<float> u(0.05f, 0.95f);
  std::vector<Setting> v;
  while (v.size() < 10) {
    const Setting s{u(rng), u(rng), u(rng)};
    const bool far = std::ranges::all_of(test, [&](const Setting& t) {
      return std::hypot(s[0] - t[0], s[1] - t[1], s[2] - t[2]) >= 0.05f;
    });
    if (off_grid(s) && far) v.push_back(s);
  }
  return v;
}

struct Protocol {
  int warm = 150, ref_frames = 300, frames = 180, first = 30;  // real warm-up and length; model frames (one shard); first second
};
Protocol protocol(sim::Effect e) { return one_shot(e) ? Protocol{1, 89, 89, 30} : Protocol{}; }

struct Reference {
  metrics::ClipStats all, first;  // the real run's statistics; one-shot effects: also its first second alone
  Clip clip;
};

Reference real_run(sim::Effect e, const Setting& s, std::uint64_t seed, bool keep_clip = false) {
  const Protocol pr = protocol(e);
  sim::Params p;
  p.effect = e;
  p.intensity = s[0];
  p.wind = s[1];
  p.turbulence = s[2];
  p.seed = seed;
  p.size = kSize;
  sim::Fluid f(p);
  for (int i = 0; i < pr.warm; ++i) f.step_frame();
  Clip cl;
  cl.allocate(kSize, pr.ref_frames);
  cl.fps = 30.f;
  for (int i = 0; i < pr.ref_frames; ++i) {
    f.step_frame();
    f.render(cl.frame(i));
  }
  Reference r;
  r.all = metrics::stats(cl);
  r.first = one_shot(e) ? metrics::stats(slice_clip(cl, 0, pr.first)) : r.all;
  if (keep_clip) r.clip = std::move(cl);
  return r;
}

int nearest_start(const rollout::Model& M, std::span<const float> ctl) {
  int best = 0;
  float bd = std::numeric_limits<float>::infinity();
  for (std::size_t k = 0; k < M.starts.size(); ++k) {
    float d = 0.f;
    for (std::size_t q = 0; q < 3 && q < M.starts[k].controls.size(); ++q) d += (M.starts[k].controls[q] - ctl[q]) * (M.starts[k].controls[q] - ctl[q]);
    if (d < bd) {
      bd = d;
      best = static_cast<int>(k);
    }
  }
  return best;
}

// How a run is started: as v1 starts (warm-up or stored fine fields), or cold (from the coarse state, no warm-up).
enum class StartMode { usual, cold };

// `frames` frames of an effect through the reference from the start point nearest the controls: v1 (mix == nullptr) or
// DCM-fine with generation options g.
Clip model_clip(const rollout::Model& M, const fine::Mixer* mix, const fine::GenOptions& g, const Setting& ctl_a, std::uint64_t seed, int frames,
                StartMode mode = StartMode::usual) {
  const std::vector<float> ctl(ctl_a.begin(), ctl_a.end());
  const int idx = nearest_start(M, ctl);
  Clip c;
  c.allocate(kSize, frames);
  c.fps = M.fps;
  std::vector<float> rgba(sz(kSize) * sz(kSize) * 4);
  fine::Frame f;
  const fine::Mixer v1 = mix ? fine::Mixer{} : fine::v1_mixer(fine::Spec{});
  const bool cold = mode == StartMode::cold;
  rollout::State s = !mix && !cold ? rollout::start(M, idx, kSize, ctl, seed)
                                   : fine::start(M, mix ? *mix : v1, idx, kSize, ctl, seed, g, cold ? 0 : -1, !cold);
  for (int i = 0; i < frames; ++i) {
    if (mix) {
      fine::step(M, *mix, s, ctl, seed, g, f);
    } else {
      rollout::step(M, s, ctl, seed);
    }
    fine::render(M, s, rgba);
    auto out = c.frame(i);
    for (std::size_t k = 0; k < rgba.size(); ++k) out[k] = static_cast<std::uint8_t>(rgba[k] * 255.f + 0.5f);
  }
  return c;
}

// rollout_train.cpp's calibration score: detail spectrum, |log motion ratio|, and the log ratios of mean light (effects
// that give light) and mean cover.
double detail_score(const metrics::ClipStats& ref, const metrics::ClipStats& test) {
  const metrics::StatDistance d = metrics::distance(ref, test);
  const auto mean = [](const std::vector<double>& v) {
    double s = 0;
    for (const double x : v) s += x;
    return v.empty() ? 0.0 : s / static_cast<double>(v.size());
  };
  const auto log_ratio = [](double a, double b) { return std::abs(std::log(std::max(1e-6, a) / std::max(1e-6, b))); };
  const double em = mean(ref.emission);
  return d.spectrum_l1 + std::abs(std::log(std::max(1e-3, d.motion_ratio))) + (em > 1e-3 ? log_ratio(mean(test.emission), em) : 0.0) +
         log_ratio(mean(test.coverage), mean(ref.coverage));
}

struct GenScore {
  double score = 0;
  metrics::StatDistance d;
};

GenScore gen_score(const metrics::ClipStats& ref, const metrics::ClipStats& test) { return {detail_score(ref, test), metrics::distance(ref, test)}; }

std::string gen_cells(const GenScore& g) {
  return std::format("{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.3f}", g.score, g.d.spectrum_l1, g.d.motion_ratio, g.d.coverage_l1, g.d.emission_l1, g.d.mean_frame_psnr);
}

// Gathers a config's inputs and context slots from every expert and context (as fine::Mixer::predict does).
void gather(const fine::Config& c, std::span<const double> x, std::span<const int> ctx, std::vector<double>& xs, std::vector<int>& cs) {
  xs.clear();
  for (const int e : c.experts) xs.push_back(x[sz(e)]);
  cs.assign(c.mixer_contexts.size() + 4, 0);
  for (std::size_t j = 0; j < c.mixer_contexts.size(); ++j) cs[j + 1] = ctx[sz(c.mixer_contexts[j])];
  const std::size_t m = c.mixer_contexts.size() + 1;
  cs[m + 1] = c.avm_context >= 0 ? ctx[sz(c.avm_context)] : 0;
  cs[m + 2] = c.scale_context >= 0 ? ctx[sz(c.scale_context)] : 0;
}

// The own-rollout pass: from each window's true fine fields, the mixer's own output fed back for kWindowSteps frames
// (the v1 stepper teacher-forced on the true run), rows sampled as in record; then one more pass of online learning.
dcm::ValueNet own_rollout_pass(const rollout::Model& M, const fine::Mixer& mix, const dcm::ValueNet& net1, const std::vector<Window>& ws,
                               const Table& train, const Prepared& one_step, int threads, std::uint64_t seed, int* n_rows = nullptr) {
  const fine::Spec& sp = mix.spec;
  struct Sample {
    std::array<double, fine::kExperts> x;
    std::array<int, fine::kContexts> ctx;
    std::array<double, fine::kScaleFeatures> z;
    double y;
  };
  std::vector<std::vector<Sample>> per(ws.size());
  const fine::GenOptions g{1.0, true, {}};
  parallel(static_cast<int>(ws.size()), threads, [&](int wi) {
    const Window& w = ws[sz(wi)];
    const RunInfo& run = train.runs.at(sz(w.run));
    const std::vector<float> ctl(run.controls.begin(), run.controls.end());
    rollout::State s;
    s.res = M.h.res;
    s.size = kSize;
    s.fine_t = w.fine_t;
    s.fine_d = w.fine_d;
    s.pressure.assign(sz(M.h.res) * sz(M.h.res), 0.f);
    fine::Frame f;
    std::mt19937_64 rng(run.seed * 7919ULL + static_cast<std::uint64_t>(w.first));
    std::vector<fine::Row> rows;
    for (std::size_t j = 0; j < w.steps.size(); ++j) {
      const WindowStep& st = w.steps[j];
      s.coarse = st.coarse;
      s.flow = st.flow;
      s.time = st.time;
      s.since_start = st.since;
      if (j > 0) {  // the rows of the first step are the one-step rows (true fine fields): the pass trains on its own
        fine::compute_frame(M, s, run.seed, ctl, f);
        rows.clear();
        sample_rows(sp, f, s, 0, st.target_t, 32, 4, rng, w.run, w.first + static_cast<int>(j), rows);
        sample_rows(sp, f, s, 1, st.target_d, 32, 4, rng, w.run, w.first + static_cast<int>(j), rows);
        for (const fine::Row& r : rows) {
          Sample smp{};
          smp.ctx = fine::contexts_of(sp, r, ctl, kSize);
          fine::normalise(sp, r, smp.x, smp.z);
          smp.y = fine::normalise_value(sp, r.channel, dbl(r.target), dbl(r.e[fine::kCoarse]));
          per[sz(wi)].push_back(smp);
        }
      }
      fine::detail_step(M, mix, s, run.seed, ctl, g, f);
    }
  });
  std::vector<Sample> all;
  for (auto& v : per) all.insert(all.end(), v.begin(), v.end());
  if (n_rows) *n_rows = static_cast<int>(all.size());
  // Continued online learning (DAgger's aggregation): one seeded pass over the own-rollout rows mixed with as many
  // one-step rows, at the rates training ended with (annealed per context and halved per pass), so the net moves gently
  // from where pass 1 left it.
  dcm::ValueNet net = net1;
  net.freeze(false);
  std::mt19937_64 rng(seed + 99);
  std::vector<std::size_t> anchor(one_step.x.size());
  std::iota(anchor.begin(), anchor.end(), std::size_t{0});
  std::shuffle(anchor.begin(), anchor.end(), rng);
  anchor.resize(std::min(anchor.size(), all.size()));
  for (const std::size_t i : anchor) all.push_back({one_step.x[i], one_step.ctx[i], one_step.z[i], one_step.y[i]});
  std::vector<std::size_t> order(all.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::shuffle(order.begin(), order.end(), rng);
  std::vector<double> xs;
  std::vector<int> cs;
  for (const std::size_t i : order) {
    gather(mix.config, all[i].x, all[i].ctx, xs, cs);
    (void)net.predict(xs, cs, all[i].z);
    net.update(all[i].y);
  }
  net.freeze();
  {  // the Laplace negative log-likelihood (nats) of the pass's rows, before and after it
    const auto nll = [&](dcm::ValueNet& n) {
      double sum = 0;
      for (const auto& smp : all) {
        gather(mix.config, smp.x, smp.ctx, xs, cs);
        const auto p = n.predict(xs, cs, smp.z);
        sum += std::abs(smp.y - p.mu) / p.b + std::log(2.0 * p.b);
      }
      return sum / static_cast<double>(std::max<std::size_t>(1, all.size()));
    };
    dcm::ValueNet before = net1;
    log(std::format("own-rollout pass: {} rows, Laplace NLL {:.4f} before, {:.4f} after", all.size(), nll(before), nll(net)));
  }
  return net;
}

struct GenOption {
  double tau;
  bool relock;
};

fs::path released_path(const Ctx& c, sim::Effect e) { return c.data / std::format("{}_released.mixer", ename(e)); }
fs::path released_gen_path(const Ctx& c, sim::Effect e) { return c.data / std::format("{}_released.gen", ename(e)); }

}  // namespace

// --- train-fine: own-rollout pass, generation calibration, re-ranking, release ----------------------------------

void train(const Ctx& c, sim::Effect e) {
  const auto t_all = std::chrono::steady_clock::now();
  const rollout::Model M = load_v1(c, e);
  const Table train = read_rows(c, e, "train", M), val = read_rows(c, e, "val", M);
  const fine::Domain d = parse_domain(c.domain);
  const fine::Spec sp = get_spec(c, e, d, M, train);
  const auto sites = run_sites(train);
  const Prepared ptr = prepare(sp, train), pv = prepare(sp, val);
  const fine::Family fam = parse_family(c.family);
  const Problem P = make_problem(sp, ptr, train, fine::family_contexts(fam, one_shot(e)), sites, 0);
  dcm::SearchSpace space;
  std::vector<dcm::SearchConfig> top;
  {
    std::ifstream in(top_path(c, e, c.family, c.domain, c.seed));
    if (!in) throw std::runtime_error("no search result for this effect (run search-fine)");
    for (std::string line; std::getline(in, line);) {
      if (!line.empty()) top.push_back(decode(line));
    }
  }
  const int n_top = c.quick ? std::min<int>(2, static_cast<int>(top.size())) : static_cast<int>(top.size());
  const std::vector<Window> ws = read_windows(windows_path(c, e));
  const auto settings = validation_settings();
  const int n_set = c.quick ? 3 : static_cast<int>(settings.size());
  const Protocol pr = protocol(e);
  const int frames = c.quick ? std::min(60, pr.frames) : pr.frames;
  // real runs at the validation settings (seeds 800000 + i), once
  std::vector<Reference> refs(sz(n_set));
  parallel(n_set, c.threads, [&](int i) { refs[sz(i)] = real_run(e, settings[sz(i)], 800000 + static_cast<std::uint64_t>(i)); });
  log(std::format("train {}: {} validation references in {:.0f} s", ename(e), n_set, seconds_since(t_all)));
  std::vector<std::string> rows;
  // statistics of one generator over the validation settings (model seeds 820000 + i)
  const auto evaluate = [&](const fine::Mixer* mix, const fine::GenOptions& g, const std::string& name) {
    std::vector<GenScore> sc(sz(n_set));
    parallel(n_set, c.threads, [&](int i) {
      const Clip cl = model_clip(M, mix, g, settings[sz(i)], 820000 + static_cast<std::uint64_t>(i), frames);
      sc[sz(i)] = gen_score(refs[sz(i)].all, metrics::stats(cl));
    });
    double mean = 0;
    for (int i = 0; i < n_set; ++i) {
      mean += sc[sz(i)].score / n_set;
      rows.push_back(std::format("{},{},{:.2f},{},{},{}", ename(e), name, g.tau, g.relock ? 1 : 0, i, gen_cells(sc[sz(i)])));
    }
    return mean;
  };
  const auto t0 = std::chrono::steady_clock::now();
  const double v1_score = evaluate(nullptr, {}, "v1");
  log(std::format("train {}: v1 validation score {:.4f} ({:.0f} s)", ename(e), v1_score, seconds_since(t0)));
  // every candidate: trained on every training row, then the own-rollout pass
  struct Candidate {
    fine::Mixer one, two;
    double bits_one = 0, bits_two = 0;
  };
  std::vector<std::string> cand_rows;
  std::vector<Candidate> cand(sz(n_top));
  for (int k = 0; k < n_top; ++k) {
    const auto tk = std::chrono::steady_clock::now();
    dcm::ValueNet net1(1, 0, dcm::ValueNetSpec{{1}});
    Candidate& cd = cand[sz(k)];
    cd.one = train_mixer(sp, P, space, top[sz(k)], c.seed, &net1);
    int n2 = 0;
    const dcm::ValueNet net2 = own_rollout_pass(M, cd.one, net1, ws, train, ptr, c.threads, c.seed, &n2);
    cd.two = fine::make_mixer(sp, cd.one.config, net2);
    cd.bits_one = mean_of(per_run(val, score_rows(cd.one, pv, val)).active);
    cd.bits_two = mean_of(per_run(val, score_rows(cd.two, pv, val)).active);
    cand_rows.push_back(std::format("{},{},{:.5f},{:.5f},{},{},\"{}\"", ename(e), k, cd.bits_one, cd.bits_two, n2, cd.two.version(), fine::describe(cd.one.config)));
    log(std::format("train {}: candidate {} {}: validation bits {:.4f} after training, {:.4f} after the own-rollout pass ({} rows), {:.0f} s", ename(e), k,
                    fine::describe(cd.one.config), cd.bits_one, cd.bits_two, n2, seconds_since(tk)));
  }
  // tau and relock on the first candidate (both passes)
  const std::vector<GenOption> grid = c.quick ? std::vector<GenOption>{{0.0, false}, {1.0, true}}
                                              : std::vector<GenOption>{{0.0, false}, {0.5, false}, {1.0, false}, {1.5, false},
                                                                       {0.0, true}, {0.5, true}, {1.0, true}, {1.5, true}};
  GenOption best_g{0.0, false};
  double best = std::numeric_limits<double>::infinity();
  bool best_two = true;
  for (const bool two : {false, true}) {
    for (const GenOption& go : grid) {
      const double s = evaluate(two ? &cand[0].two : &cand[0].one, {go.tau, go.relock, {}}, std::format("cand0_pass{}", two ? 2 : 1));
      log(std::format("train {}: candidate 0 pass {} tau {} relock {}: score {:.4f} (v1 {:.4f})", ename(e), two ? 2 : 1, go.tau, go.relock, s, v1_score));
      if (s < best) {
        best = s;
        best_g = go;
        best_two = two;
      }
    }
  }
  // re-rank the candidates at that setting
  int winner = 0;
  double win_score = best;
  for (int k = 1; k < n_top; ++k) {
    const fine::Mixer& m = best_two ? cand[sz(k)].two : cand[sz(k)].one;
    const double s = evaluate(&m, {best_g.tau, best_g.relock, {}}, std::format("cand{}_pass{}", k, best_two ? 2 : 1));
    log(std::format("train {}: candidate {} at tau {} relock {}: score {:.4f}", ename(e), k, best_g.tau, best_g.relock, s));
    if (s < win_score) {
      win_score = s;
      winner = k;
    }
  }
  if (winner != 0) {  // its own tau and relock
    for (const GenOption& go : grid) {
      if (go.tau == best_g.tau && go.relock == best_g.relock) continue;
      const fine::Mixer& m = best_two ? cand[sz(winner)].two : cand[sz(winner)].one;
      const double s = evaluate(&m, {go.tau, go.relock, {}}, std::format("cand{}_pass{}", winner, best_two ? 2 : 1));
      if (s < win_score) {
        win_score = s;
        best_g = go;
      }
    }
  }
  const fine::Mixer& rel = best_two ? cand[sz(winner)].two : cand[sz(winner)].one;
  fine::save_mixer(released_path(c, e), rel);
  std::ofstream(released_gen_path(c, e)) << std::format("{} {} {} {} {}\n", best_g.tau, best_g.relock ? 1 : 0, winner, best_two ? 2 : 1, rel.version());
  merge_csv(c.results / "g_fine_val.csv", "effect,candidate,tau,relock,setting,score,spectrum_l1,motion_ratio,coverage_l1,emission_l1,mean_frame_psnr", rows);
  merge_csv(c.results / "g_fine_candidates.csv", "effect,candidate,val_bits_trained,val_bits_own_rollout,own_rollout_rows,version,config", cand_rows);
  log(std::format("train {}: released candidate {} (pass {}), tau {}, relock {}: validation score {:.4f} against v1 {:.4f}; {}; version {}; {:.0f} min",
                  ename(e), winner, best_two ? 2 : 1, best_g.tau, best_g.relock, win_score, v1_score, fine::describe(rel.config), rel.version(),
                  seconds_since(t_all) / 60.0));
}
// --- eval-fine: G1c and the test, once ---------------------------------------------------------------------------

namespace {

double apsnr(std::span<const std::uint8_t> r, std::span<const std::uint8_t> t) {
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

const std::vector<int> kHorizons = {1, 4, 8, 16, 30, 60, 120, 240};

}  // namespace

void eval(const Ctx& c, sim::Effect e) {
  const auto t_all = std::chrono::steady_clock::now();
  const rollout::Model M = load_v1(c, e);
  const fine::Mixer mix = fine::load_mixer(released_path(c, e));
  fine::GenOptions g;
  {
    std::ifstream in(released_gen_path(c, e));
    int relock = 0;
    in >> g.tau >> relock;
    if (!in) throw std::runtime_error("no released generation options (run train-fine)");
    g.relock = relock != 0;
  }
  log(std::format("eval {}: released mixer {} ({}), tau {}, relock {}", ename(e), mix.version(), fine::describe(mix.config), g.tau, g.relock));
  const Protocol pr = protocol(e);
  const auto settings = test_settings();
  const int n_set = c.quick ? 2 : static_cast<int>(settings.size());
  const int frames = c.quick ? std::min(60, pr.frames) : pr.frames;
  // endless statistics at the held-out settings: real runs (seeds 900000 + i, as d-eval) against single shards from the
  // nearest start point (seeds 920000 + i), v1 and DCM-fine through the reference; G1c: the first second of a cold start
  struct SetResult {
    std::map<std::string, GenScore> d;
    Clip real, v1, dcm;
  };
  std::vector<SetResult> res(sz(n_set));
  parallel(n_set, c.threads, [&](int i) {
    const Setting& s = settings[sz(i)];
    SetResult& r = res[sz(i)];
    Reference ref = real_run(e, s, 900000 + static_cast<std::uint64_t>(i), i == 0);
    const Reference other = real_run(e, s, 910000 + static_cast<std::uint64_t>(i));
    r.d["real_other_seed"] = gen_score(ref.all, other.all);
    const std::uint64_t seed = 920000 + static_cast<std::uint64_t>(i);
    Clip v1 = model_clip(M, nullptr, {}, s, seed, frames);
    Clip dc = model_clip(M, &mix, g, s, seed, frames);
    r.d["v1"] = gen_score(ref.all, metrics::stats(v1));
    r.d["dcm_fine"] = gen_score(ref.all, metrics::stats(dc));
    // G1c: the first second (30 frames) of the current start (v1) and of cold starts, against the real run
    const auto first = [&](const Clip& cl) { return metrics::stats(slice_clip(cl, 0, pr.first)); };
    r.d["g1c_v1_usual"] = gen_score(ref.first, first(v1));
    r.d["g1c_dcm_usual"] = gen_score(ref.first, first(dc));
    r.d["g1c_v1_cold"] = gen_score(ref.first, metrics::stats(model_clip(M, nullptr, {}, s, seed, pr.first, StartMode::cold)));
    r.d["g1c_dcm_cold"] = gen_score(ref.first, metrics::stats(model_clip(M, &mix, g, s, seed, pr.first, StartMode::cold)));
    if (i == 0) {
      r.real = std::move(ref.clip);
      r.v1 = std::move(v1);
      r.dcm = std::move(dc);
    }
    log(std::format("eval {}: setting {} done ({:.0f} s)", ename(e), i, seconds_since(t_all)));
  });
  std::vector<std::string> stat_rows;
  for (int i = 0; i < n_set; ++i) {
    for (const auto& [name, gs] : res[sz(i)].d) stat_rows.push_back(std::format("{},{},{},{}", ename(e), i, name, gen_cells(gs)));
  }
  merge_csv(c.results / "g_fine_test_stats.csv", "effect,setting,method,score,spectrum_l1,motion_ratio,coverage_l1,emission_l1,mean_frame_psnr", stat_rows);
  if (e == sim::Effect::fire || c.quick) {  // figure: real, v1, DCM-fine at the first held-out setting across the shard
    std::vector<int> cols = {0, 29, 59, 89, 119, 149, std::min(179, frames - 1)};
    if (one_shot(e) || frames < 180) cols = {0, frames / 6, frames / 3, frames / 2, 2 * frames / 3, 5 * frames / 6, frames - 1};
    const auto pick = [&](const Clip& src) {
      Clip o;
      o.allocate(kSize, static_cast<int>(cols.size()));
      for (std::size_t k = 0; k < cols.size(); ++k) {
        const auto from = src.frame(std::min(cols[k], src.frames - 1));
        const auto to = o.frame(static_cast<int>(k));
        std::copy(from.begin(), from.end(), to.begin());
      }
      return o;
    };
    const Clip a = pick(res[0].real), b = pick(res[0].v1), cc = pick(res[0].dcm);
    std::vector<const Clip*> sheet = {&a, &b, &cc};
    fs::create_directories(c.figures);
    if (auto w = write_png(c.figures / std::format("g_fine_endless{}.png", e == sim::Effect::fire ? "" : "_" + ename(e)),
                           comparison_sheet(sheet, static_cast<int>(cols.size()), Background::black));
        !w) {
      log("figure: " + w.error());
    }
  }
  // tracking: 8 held-out runs (salt 2) from their true state with their own seed, as d-eval
  const int n_runs = c.quick ? 2 : 8, KE = one_shot(e) ? 89 : (c.quick ? 60 : 240), warm = one_shot(e) ? 1 : 100;
  std::vector<std::array<std::vector<double>, 2>> curves(sz(n_runs));
  parallel(n_runs, c.threads, [&](int r) {
    const sim::Params p = run_params(e, 2, static_cast<std::uint64_t>(r));
    sim::Fluid truth(p);
    for (int i = 0; i < warm; ++i) truth.step_frame();
    const sim::State st = truth.state();
    rollout::Model mt = M;
    mt.h.start_fine = kSize;
    rollout::StartPoint sp;
    sp.controls = {p.intensity, p.wind, p.turbulence};
    sp.seed = p.seed;
    sp.time = st.time;
    sp.coarse.resize(sz(mt.h.res) * sz(mt.h.res) * rollout::kPhys);
    rollout::coarse_from_sim(st, mt.h.res, p.fps, sp.coarse);
    sp.fine_t = st.temp;
    sp.fine_d = st.soot;
    mt.starts = {sp};
    rollout::State a = rollout::start(mt, 0, kSize, sp.controls, sp.seed);
    rollout::State b = fine::start(mt, mix, 0, kSize, sp.controls, sp.seed, g);
    fine::Frame f;
    std::vector<float> rgba(sz(kSize) * sz(kSize) * 4);
    std::vector<std::uint8_t> ref(rgba.size()), out(rgba.size());
    for (int i = 0; i < KE; ++i) {
      truth.step_frame();
      truth.render(ref);
      rollout::step(mt, a, sp.controls, sp.seed);
      fine::render(mt, a, rgba);
      for (std::size_t k = 0; k < rgba.size(); ++k) out[k] = static_cast<std::uint8_t>(rgba[k] * 255.f + 0.5f);
      curves[sz(r)][0].push_back(apsnr(ref, out));
      fine::step(mt, mix, b, sp.controls, sp.seed, g, f);
      fine::render(mt, b, rgba);
      for (std::size_t k = 0; k < rgba.size(); ++k) out[k] = static_cast<std::uint8_t>(rgba[k] * 255.f + 0.5f);
      curves[sz(r)][1].push_back(apsnr(ref, out));
    }
    log(std::format("eval {}: tracking run {} done ({:.0f} s)", ename(e), r, seconds_since(t_all)));
  });
  std::vector<std::string> track_rows;
  for (int r = 0; r < n_runs; ++r) {
    for (int m = 0; m < 2; ++m) {
      std::string row = std::format("{},{},{}", ename(e), r, m == 0 ? "v1" : "dcm_fine");
      for (const int h : kHorizons) row += h <= KE ? std::format(",{:.3f}", curves[sz(r)][sz(m)][sz(h - 1)]) : ",";
      track_rows.push_back(row);
    }
  }
  std::string th = "effect,run,method";
  for (const int h : kHorizons) th += std::format(",f{}", h);
  merge_csv(c.results / "g_fine_test_track.csv", th, track_rows);
  log(std::format("eval {}: done in {:.1f} min", ename(e), seconds_since(t_all) / 60.0));
}
// --- bench-experts: what each part costs per pixel ---------------------------------------------------------------
//
// The reference (fine.cpp) builds a Row per pixel and is far too slow to say what DCM-fine would cost in the runtime.
// This times a plain, allocation-free float version of each part of the pass on frames of a v1 rollout (scalar C++,
// baseline ISA: an upper bound for the runtime's SIMD code). v1's own buffers (the advected fields, the lock and its
// parts, the flicker octaves) are given: v1 computes them anyway. Each part is timed over the frames' evaluated pixels
// (not skipped) or whole planes, and converted to ms per 128 x 128 frame.

namespace {

struct Planes {
  int S = 0, R = 0;
  std::array<std::vector<float>, 2> A, Asl, L, rA, aup, prev, cup, bres, lap, grad, adiff, n1, n2, n3;
  std::array<std::vector<float>, 3> oct;  // raw flicker octaves where there is new material
  std::vector<float> ux, vy, speed, vortc, vort, coarse, flow, grain;
  std::array<std::vector<std::uint8_t>, 2> lvl, ratio, flowc, macro;
  std::vector<std::uint8_t> cellmacro;
  std::array<std::vector<int>, 2> eval;  // evaluated (not skipped) pixels per channel
  std::array<std::vector<float>, 2> xs;  // per evaluated pixel: its 15 experts side by side (as a fused pass has them)
  std::array<std::vector<float>, 2> out;
};

// Separable bilinear upsampling of a coarse field (stride `ch`, channel c) to S x S, as the runtime does it: the
// interpolation indices and weights along an axis are fixed per size.
struct Axis {
  std::vector<int> i;
  std::vector<float> w;
  Axis(int R, int S) {
    const float k = fl(S) / fl(R);
    for (int x = 0; x < S; ++x) {
      const float xc = std::clamp((fl(x) + 0.5f) / k - 0.5f, 0.f, fl(R - 1));
      const int x0 = std::min(static_cast<int>(xc), R - 2);
      i.push_back(x0);
      w.push_back(xc - fl(x0));
    }
  }
};

void upsample(const Axis& ax, const float* f, int R, int ch, int c, int S, float* out, std::vector<float>& row) {
  row.resize(sz(R));
  for (int y = 0; y < S; ++y) {
    const int y0 = ax.i[sz(y)];
    const float wy = ax.w[sz(y)];
    for (int i = 0; i < R; ++i) {
      const float a = f[(sz(y0) * sz(R) + sz(i)) * sz(ch) + sz(c)], b = f[(sz(y0 + 1) * sz(R) + sz(i)) * sz(ch) + sz(c)];
      row[sz(i)] = a + wy * (b - a);
    }
    float* o = out + sz(y) * sz(S);
    for (int x = 0; x < S; ++x) {
      const int x0 = ax.i[sz(x)];
      o[x] = row[sz(x0)] + ax.w[sz(x)] * (row[sz(x0 + 1)] - row[sz(x0)]);
    }
  }
}

// Value-noise lattice of one octave for a frame (both time slices), then smooth interpolation per pixel: the grain.
struct GrainLattice {
  int n = 0;
  float freq = 0, tz = 0;
  std::vector<float> a, b;
  void fill(std::uint64_t seed, float f, float z, int S) {
    freq = f;
    n = static_cast<int>(std::ceil(fl(S) * f)) + 3;
    a.resize(sz(n) * sz(n));
    b.resize(a.size());
    const float fz = std::floor(z);
    tz = smooth5(z - fz);
    const auto iz = static_cast<std::int32_t>(fz);
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        a[sz(j) * sz(n) + sz(i)] = cell_value(i, j, iz, seed);
        b[sz(j) * sz(n) + sz(i)] = cell_value(i, j, iz + 1, seed);
      }
    }
  }
  float value(float x, float y) const {
    x *= freq;
    y *= freq;
    const int ix = static_cast<int>(x), iy = static_cast<int>(y);
    const float tx = smooth5(x - fl(ix)), ty = smooth5(y - fl(iy));
    const std::size_t k = sz(iy) * sz(n) + sz(ix), nn = sz(n);
    const auto lerp = [](float p, float q, float t) { return p + t * (q - p); };
    const float a0 = lerp(lerp(a[k], a[k + 1], tx), lerp(a[k + nn], a[k + nn + 1], tx), ty);
    const float a1 = lerp(lerp(b[k], b[k + 1], tx), lerp(b[k + nn], b[k + nn + 1], tx), ty);
    return lerp(a0, a1, tz);
  }
};

// Microseconds of the calling thread's CPU time, the least of `reps` repetitions (time spent waiting for a busy machine's cores does not count).
double thread_us() {
  timespec ts{};
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
  return static_cast<double>(ts.tv_sec) * 1e6 + static_cast<double>(ts.tv_nsec) * 1e-3;
}

template <class F>
double time_us(F&& fn, int reps) {
  std::vector<double> v;
  for (int r = 0; r < reps; ++r) {
    const double t0 = thread_us();
    fn();
    v.push_back(thread_us() - t0);
  }
  return std::ranges::min(v);  // the least disturbed repetition (other jobs only ever add time)
}

double load_average() {
  std::ifstream in("/proc/loadavg");
  double l = -1;
  in >> l;
  return l;
}

}  // namespace

void bench(const Ctx& c, sim::Effect e) {
  const rollout::Model M = load_v1(c, e);
  const Table train = read_rows(c, e, "train", M);
  const fine::Spec sp = get_spec(c, e, fine::Domain::linear, M, train);
  const int S = kSize, R = M.h.res, C = M.h.channels();
  const std::size_t S2 = sz(S) * sz(S), N = sz(R) * sz(R);
  // frames of a v1 rollout from start point 0 (the reference), every 4th of 120 after 30
  std::vector<Planes> frames;
  {
    const std::vector<float> ctl{0.6f, 0.5f, 0.6f};
    rollout::State s = rollout::start(M, 0, S, ctl, 4242);
    fine::Frame f;
    for (int k = 0; k < (c.quick ? 40 : 150); ++k) {
      rollout::State before = s;
      rollout::step(M, s, ctl, 4242);
      if (k < 30 || k % 4) continue;
      // the frame's buffers: v1's step from `before` with the stepped coarse state
      before.coarse = s.coarse;
      before.flow = s.flow;
      before.pressure = s.pressure;
      fine::compute_frame(M, before, 4242, ctl, f);
      Planes p;
      p.S = S;
      p.R = R;
      for (int q = 0; q < 2; ++q) {
        p.A[sz(q)] = f.A[sz(q)];
        p.Asl[sz(q)] = f.Asl[sz(q)];
        p.L[sz(q)] = f.L[sz(q)];
        p.rA[sz(q)] = f.rA[sz(q)];
        p.aup[sz(q)] = f.aup[sz(q)];
        p.prev[sz(q)] = f.prev[sz(q)];
        for (auto* v : {&p.cup, &p.bres, &p.lap, &p.grad, &p.adiff, &p.n1, &p.n2, &p.n3, &p.out}) (*v)[sz(q)].assign(S2, 0.f);
        for (auto* v : {&p.lvl, &p.ratio, &p.flowc, &p.macro}) (*v)[sz(q)].assign(S2, 0);
        for (std::size_t i = 0; i < S2; ++i) {
          if (!fine::skip_pixel(sp, f, i, q)) p.eval[sz(q)].push_back(static_cast<int>(i));
        }
      }
      for (int o = 0; o < 3; ++o) {
        p.oct[sz(o)].assign(S2, 0.f);
        const auto& n = o == 0 ? f.n1[0] : o == 1 ? f.n2[0] : f.n3[0];
        for (std::size_t i = 0; i < S2; ++i) p.oct[sz(o)][i] = f.aup[0][i] > 0.f ? n[i] / f.aup[0][i] : 0.f;
      }
      p.ux = f.ux;
      p.vy = f.vy;
      p.speed.assign(S2, 0.f);
      p.vort.assign(S2, 0.f);
      p.vortc.assign(N, 0.f);
      p.grain.assign(S2, 0.f);
      p.coarse = s.coarse;
      p.flow = s.flow;
      p.cellmacro.assign(N, 0);
      frames.push_back(std::move(p));
    }
  }
  const auto n_frames = static_cast<double>(frames.size());
  double evaluated = 0;
  for (const auto& p : frames) evaluated += static_cast<double>(p.eval[0].size() + p.eval[1].size()) / n_frames;
  const int reps = c.quick ? 3 : 15;
  std::vector<float> row;
  const Axis axis(R, S);
  const auto over_frames = [&](auto&& fn) {
    return time_us([&] {
      for (auto& p : frames) fn(p);
    }, reps) / n_frames;  // microseconds per frame
  };
  std::vector<std::pair<std::string, double>> parts;  // name, us per frame
  // experts
  parts.emplace_back("group_adv", over_frames([&](Planes& p) {
    for (int q = 0; q < 2; ++q) {
      for (std::size_t i = 0; i < S2; ++i) p.adiff[sz(q)][i] = p.A[sz(q)][i] - p.Asl[sz(q)][i];
    }
  }));
  parts.emplace_back("group_lock", 0.0);  // L, r_up A and a_up: v1's lock computes them
  parts.emplace_back("group_noise", over_frames([&](Planes& p) {
    for (int q = 0; q < 2; ++q) {
      const float* a = p.aup[sz(q)].data();
      for (std::size_t i = 0; i < S2; ++i) {
        p.n1[sz(q)][i] = a[i] * p.oct[0][i];
        p.n2[sz(q)][i] = a[i] * p.oct[1][i];
        p.n3[sz(q)][i] = a[i] * p.oct[2][i];
      }
    }
  }));
  parts.emplace_back("group_coarse", over_frames([&](Planes& p) {
    std::vector<float> res(N);
    for (int q = 0; q < 2; ++q) {
      upsample(axis, p.coarse.data(), R, C, 2 + q, S, p.cup[sz(q)].data(), row);
      // block residual: the lock's block means are v1's; C - mean per cell, upsampled
      for (std::size_t j = 0; j < N; ++j) res[j] = p.coarse[j * sz(C) + 2 + sz(q)] - 0.5f;
      upsample(axis, res.data(), R, 1, 0, S, p.bres[sz(q)].data(), row);
    }
  }));
  parts.emplace_back("group_shape", over_frames([&](Planes& p) {
    for (int q = 0; q < 2; ++q) {
      const float* a = p.A[sz(q)].data();
      for (int y = 1; y < S - 1; ++y) {
        for (int x = 1; x < S - 1; ++x) {
          const std::size_t i = sz(y) * sz(S) + sz(x);
          const float l = a[i - 1], r = a[i + 1], d = a[i - sz(S)], u = a[i + sz(S)];
          p.lap[sz(q)][i] = l + r + d + u - 4.f * a[i];
          const float gx = 0.5f * (r - l), gy = 0.5f * (u - d);
          p.grad[sz(q)][i] = std::sqrt(gx * gx + gy * gy) * a[i];
        }
      }
    }
  }));
  parts.emplace_back("group_prev", over_frames([&](Planes& p) {
    for (int q = 0; q < 2; ++q) std::copy(p.A[sz(q)].begin(), p.A[sz(q)].end(), p.prev[sz(q)].begin());
  }));
  parts.emplace_back("group_bias", 0.0);
  // contexts
  parts.emplace_back("context_lvl", over_frames([&](Planes& p) {
    for (int q = 0; q < 2; ++q) {
      const float inv = 1.f / sp.s[sz(q)];
      for (const int i : p.eval[sz(q)]) {
        const float v = p.cup[sz(q)][sz(i)] * inv;
        p.lvl[sz(q)][sz(i)] = static_cast<std::uint8_t>(v < sp.empty ? 0 : 1 + (v >= sp.lvl[sz(q)][0]) + (v >= sp.lvl[sz(q)][1]) + (v >= sp.lvl[sz(q)][2]) + (v >= sp.lvl[sz(q)][3]));
      }
    }
  }));
  parts.emplace_back("context_ratio", over_frames([&](Planes& p) {
    for (int q = 0; q < 2; ++q) {
      for (const int i : p.eval[sz(q)]) {
        const float v = p.A[sz(q)][sz(i)] / (p.cup[sz(q)][sz(i)] + sp.eps[sz(q)]);
        p.ratio[sz(q)][sz(i)] = static_cast<std::uint8_t>((v >= sp.ratio[sz(q)][0]) + (v >= sp.ratio[sz(q)][1]) + (v >= sp.ratio[sz(q)][2]));
      }
    }
  }));
  parts.emplace_back("context_flow", over_frames([&](Planes& p) {
    const auto U = [&](int x, int y, int ch) { return p.flow[(sz(std::clamp(y, 0, R - 1)) * sz(R) + sz(std::clamp(x, 0, R - 1))) * 2 + sz(ch)]; };
    for (int y = 0; y < R; ++y) {
      for (int x = 0; x < R; ++x) p.vortc[sz(y) * sz(R) + sz(x)] = 0.5f * (U(x + 1, y, 1) - U(x - 1, y, 1)) - 0.5f * (U(x, y + 1, 0) - U(x, y - 1, 0));
    }
    upsample(axis, p.vortc.data(), R, 1, 0, S, p.vort.data(), row);
    const float inv = 128.f / fl(S);
    for (std::size_t i = 0; i < S2; ++i) {
      const float spd = std::sqrt(p.ux[i] * p.ux[i] + p.vy[i] * p.vy[i]) * inv;
      const auto f = static_cast<std::uint8_t>(((spd >= sp.speed[0]) + (spd >= sp.speed[1])) * 2 + (p.vort[i] >= 0.f));
      p.flowc[0][i] = f;
      p.flowc[1][i] = f;
    }
  }));
  for (const std::string name : {"context_height", "context_ctrl", "context_age", "context_channel", "context_extra"}) parts.emplace_back(name, 0.0);  // constants per row or frame
  for (const int K : {4, 8}) {
    const std::vector<double>& cen = K == 4 ? sp.macro4 : sp.macro8;
    parts.emplace_back(std::format("context_macro{}", K), over_frames([&](Planes& p) {
      // window means by summed-area tables of heat, soot, speed, |vorticity| on the coarse grid, nearest centroid per cell
      std::array<std::vector<float>, kMacroStatsBench> sat;
      for (auto& t : sat) t.assign(sz(R + 1) * sz(R + 1), 0.f);
      for (int y = 0; y < R; ++y) {
        for (int x = 0; x < R; ++x) {
          const std::size_t j = sz(y) * sz(R) + sz(x);
          const float u = p.flow[j * 2], v = p.flow[j * 2 + 1];
          const std::array<float, 4> val{p.coarse[j * sz(C) + 2], p.coarse[j * sz(C) + 3], std::sqrt(u * u + v * v), std::abs(p.vortc[j])};
          for (int k = 0; k < 4; ++k) {
            auto& t = sat[sz(k)];
            t[sz(y + 1) * sz(R + 1) + sz(x + 1)] = val[sz(k)] + t[sz(y) * sz(R + 1) + sz(x + 1)] + t[sz(y + 1) * sz(R + 1) + sz(x)] - t[sz(y) * sz(R + 1) + sz(x)];
          }
        }
      }
      for (int y = 0; y < R; ++y) {
        for (int x = 0; x < R; ++x) {
          const int x0 = std::max(0, x - 4), x1 = std::min(R, x + 4), y0 = std::max(0, y - 4), y1 = std::min(R, y + 4);
          const float inv = 1.f / fl((x1 - x0) * (y1 - y0));
          std::array<float, 4> v{};
          for (int k = 0; k < 4; ++k) {
            const auto& t = sat[sz(k)];
            const float sum = t[sz(y1) * sz(R + 1) + sz(x1)] - t[sz(y0) * sz(R + 1) + sz(x1)] - t[sz(y1) * sz(R + 1) + sz(x0)] + t[sz(y0) * sz(R + 1) + sz(x0)];
            v[sz(k)] = (sum * inv - sp.macro_mean[sz(k)]) / sp.macro_sd[sz(k)];
          }
          int best = 0;
          float bd = 1e30f;
          for (int q = 0; q < K; ++q) {
            float d = 0;
            for (int k = 0; k < 4; ++k) {
              const float t = v[sz(k)] - static_cast<float>(cen[sz(q) * 4 + sz(k)]);
              d += t * t;
            }
            if (d < bd) {
              bd = d;
              best = q;
            }
          }
          p.cellmacro[sz(y) * sz(R) + sz(x)] = static_cast<std::uint8_t>(best);
        }
      }
      const int kk = S / R;
      for (int y = 0; y < S; ++y) {
        for (int x = 0; x < S; ++x) p.macro[0][sz(y) * sz(S) + sz(x)] = p.cellmacro[sz(y / kk) * sz(R) + sz(x / kk)];
      }
    }));
  }
  // the mixer: normalise, gather, context-selected dot products, final mixer, AVM, scale net, grain sample, clamp
  struct Weights {
    std::vector<float> w;  // [mixer][context][16]
    std::vector<float> fin, scale, avm;
  } W;
  std::mt19937 rng(1);
  std::uniform_real_distribution<float> uw(-0.1f, 0.1f);
  W.w.resize(4 * 16 * 16);
  for (auto& v : W.w) v = uw(rng);
  W.fin = {0.7f, 0.1f, 0.1f, 0.1f};
  W.scale.resize(16 * 5);
  for (auto& v : W.scale) v = uw(rng);
  W.avm.assign(16 * 33, 0.01f);
  // In the linear domain a mixer is linear in the raw experts (the bias's raw value is C_up + eps), so the dot products
  // run on raw values and one reciprocal per pixel gives the normalised mean for the AVM and the scale features. Pixels
  // go in blocks of 8 so that the arithmetic runs across pixels (as SIMD code would); context-selected weights are
  // gathered per pixel first. exp is a polynomial approximation (b only scales the grain).
  const auto fast_exp = [](float x) {
    x = std::clamp(x, -12.f, 12.f) * 1.4426950409f;
    const float fi = std::floor(x), f = x - fi;
    const float p2 = 1.f + f * (0.6931472f + f * (0.2402265f + f * (0.0555041f + f * 0.0096181f)));
    return std::bit_cast<float>(std::bit_cast<std::int32_t>(p2) + (static_cast<std::int32_t>(fi) << 23));
  };
  const auto mixer_pass = [&](int n_in, int n_mix) {
    return over_frames([&](Planes& p) {
      constexpr int Bk = 8;
      for (int q = 0; q < 2; ++q) {
        const std::array<const std::uint8_t*, 4> ctx{p.lvl[sz(q)].data(), p.lvl[sz(q)].data(), p.ratio[sz(q)].data(), p.flowc[sz(q)].data()};
        const float eps = sp.eps[sz(q)];
        const float* xs = p.xs[sz(q)].data();
        float* out = p.out[sz(q)].data();
        const auto& ev = p.eval[sz(q)];
        alignas(32) float X[16][Bk], Wsel[16][Bk], cu[Bk], inv[Bk], v[Bk], h[Bk], z0[Bk], z1[Bk], z3[Bk], gn[Bk];
        std::uint8_t cx[4][Bk];
        for (std::size_t s0 = 0; s0 < ev.size(); s0 += Bk) {
          const int cnt = static_cast<int>(std::min<std::size_t>(Bk, ev.size() - s0));
          for (int b = 0; b < Bk; ++b) {
            const std::size_t r = s0 + sz(std::min(b, cnt - 1)), i = sz(ev[r]);
            const float* xe = xs + r * 16;
            cu[b] = std::max(0.f, xe[fine::kCoarse]) + eps;
            for (int k = 0; k < n_in - 1; ++k) X[k][b] = xe[k];
            X[n_in - 1][b] = cu[b];
            for (int m = 1; m < n_mix; ++m) cx[m][b] = ctx[sz(m)][i];
            cx[1][b] = ctx[1][i];
            z0[b] = std::abs(xe[fine::kAdvDiff]);
            z1[b] = xe[fine::kGrad];
            z3[b] = xe[fine::kNew];
            gn[b] = p.grain[i];
          }
          for (int b = 0; b < Bk; ++b) {
            inv[b] = 1.f / cu[b];
            v[b] = 0.f;
          }
          for (int m = 0; m < n_mix; ++m) {
            for (int b = 0; b < Bk; ++b) h[b] = 0.f;
            if (m == 0) {
              const float* w = W.w.data();
              for (int k = 0; k < n_in; ++k) {
                for (int b = 0; b < Bk; ++b) h[b] += w[k] * X[k][b];
              }
            } else {
              for (int b = 0; b < Bk; ++b) {
                const float* w = W.w.data() + (sz(m) * 16 + sz(cx[m][b])) * 16;
                for (int k = 0; k < n_in; ++k) Wsel[k][b] = w[k];
              }
              for (int k = 0; k < n_in; ++k) {
                for (int b = 0; b < Bk; ++b) h[b] += Wsel[k][b] * X[k][b];
              }
            }
            for (int b = 0; b < Bk; ++b) v[b] += W.fin[sz(m)] * h[b];
          }
          for (int b = 0; b < cnt; ++b) {
            float mu = v[b] * inv[b];
            const float pos = std::clamp((mu + 4.f) * 4.f, 0.f, 31.999f);
            const auto j = static_cast<std::size_t>(pos);
            mu += 0.3f * (W.avm[j] + (pos - static_cast<float>(j)) * (W.avm[j + 1] - W.avm[j]));
            const float* ws = W.scale.data() + sz(cx[1][b]) * 5;
            const float lb = ws[0] + (ws[1] * z0[b] + ws[2] * z1[b] + ws[4] * z3[b] + ws[3] * (cu[b] - eps)) * inv[b];
            out[sz(ev[s0 + sz(b)])] = std::max(0.f, (mu + 0.8f * fast_exp(lb) * gn[b]) * cu[b]);
          }
        }
      }
    });
  };
  for (auto& p : frames) {
    for (int q = 0; q < 2; ++q) {
      const std::array<const float*, 15> planes{p.A[sz(q)].data(), p.Asl[sz(q)].data(), p.adiff[sz(q)].data(), p.L[sz(q)].data(), p.rA[sz(q)].data(),
                                                p.aup[sz(q)].data(), p.n1[sz(q)].data(), p.n2[sz(q)].data(), p.n3[sz(q)].data(), p.cup[sz(q)].data(),
                                                p.bres[sz(q)].data(), p.lap[sz(q)].data(), p.grad[sz(q)].data(), p.prev[sz(q)].data(), p.A[sz(q)].data()};
      p.xs[sz(q)].assign(p.eval[sz(q)].size() * 16, 0.f);
      for (std::size_t r = 0; r < p.eval[sz(q)].size(); ++r) {
        for (int k = 0; k < 15; ++k) p.xs[sz(q)][r * 16 + sz(k)] = planes[sz(k)][sz(p.eval[sz(q)][r])];
      }
    }
  }
  const double m15_1 = mixer_pass(15, 1), m15_4 = mixer_pass(15, 4), m5_1 = mixer_pass(5, 1);
  // grain: two octaves of lattice value noise per pixel, mapped to Laplace quantiles by a table
  std::vector<float> lut(1024);
  for (int k = 0; k < 1024; ++k) {  // a uniform level to a Laplace quantile (the timing does not depend on the table's values)
    const double u = (static_cast<double>(k) + 0.5) / 1024.0;
    lut[sz(k)] = static_cast<float>(u < 0.5 ? std::log(2.0 * u) : -std::log(2.0 * (1.0 - u)));
  }
  GrainLattice g0, g1;
  const double grain_us = over_frames([&](Planes& p) {
    g0.fill(77, sp.grain_freq, 0.37f * sp.grain_rate, S);
    g1.fill(78, 2.f * sp.grain_freq, 0.74f * sp.grain_rate, S);
    for (const int i : p.eval[0].size() >= p.eval[1].size() ? p.eval[0] : p.eval[1]) {  // about the union
      const float x = fl(i % S) + 0.5f, y = fl(i / S) + 0.5f;
      const float v = (g0.value(x, y) + 0.5f * g1.value(x, y)) / 1.5f;
      p.grain[sz(i)] = lut[sz(std::clamp(static_cast<int>((v + 1.f) * 512.f), 0, 1023))];
    }
  });
  const double relock_us = over_frames([&](Planes& p) {
    // block sums of the output, factors per block, and the factors interpolated at the evaluated pixels only (the others
    // hold invisible values)
    const int kk = S / R;
    std::vector<float> B(N), rr(N), aa(N);
    for (int q = 0; q < 2; ++q) {
      std::ranges::fill(B, 0.f);
      float* Q = p.out[sz(q)].data();
      for (int y = 0; y < S; ++y) {
        float* b = B.data() + sz(y / kk) * sz(R);
        const float* o = Q + sz(y) * sz(S);
        for (int x = 0; x < S; ++x) b[x / kk] += o[x];
      }
      for (std::size_t j = 0; j < N; ++j) {
        const float b = B[j] / fl(kk * kk), t = p.coarse[j * sz(C) + 2 + sz(q)];
        rr[j] = (t + 1e-4f) / (b + 1e-4f);
        aa[j] = std::max(0.f, t - b * rr[j]);
      }
      for (const int ii : p.eval[sz(q)]) {
        const auto i = sz(ii);
        const int x = ii % S, y = ii / S, x0 = axis.i[sz(x)], y0 = axis.i[sz(y)];
        const float wx = axis.w[sz(x)], wy = axis.w[sz(y)];
        const std::size_t a00 = sz(y0) * sz(R) + sz(x0);
        const float r0 = rr[a00] + wx * (rr[a00 + 1] - rr[a00]), r1 = rr[a00 + sz(R)] + wx * (rr[a00 + sz(R) + 1] - rr[a00 + sz(R)]);
        const float s0 = aa[a00] + wx * (aa[a00 + 1] - aa[a00]), s1 = aa[a00 + sz(R)] + wx * (aa[a00 + sz(R) + 1] - aa[a00 + sz(R)]);
        Q[i] = Q[i] * (r0 + wy * (r1 - r0)) + (s0 + wy * (s1 - s0));
      }
    }
  });
  const double per_input = std::max(0.0, (m15_1 - m5_1) / 10.0), per_mixer = std::max(0.0, (m15_4 - m15_1) / 3.0);
  const double fixed = std::max(0.0, m5_1 - 5.0 * per_input);
  const double base_us = fixed + grain_us + relock_us;
  const double load = load_average();
  // cost model: base (mixer overhead without inputs, grain, relock), each group its experts plus its inputs in the
  // context-free mixer, each context its own computation plus one more mixer over all inputs (conservative)
  std::vector<std::string> rows;
  const auto add = [&](const std::string& name, double us_frame, const std::string& note) {
    rows.push_back(std::format("{},{},{:.4f},{:.4f},{:.0f},{:.2f},{}", ename(e), name, us_frame / (static_cast<double>(S2) * 2.0), us_frame / 1000.0,
                               evaluated, load, note));
  };
  add("base", base_us, "mixer without inputs, grain, relock");
  for (const auto& gr : fine::expert_groups()) {
    double us = per_input * static_cast<double>(gr.experts.size());
    for (const auto& [n, v] : parts) {
      if (n == std::format("group_{}", gr.name)) us += v;
    }
    add(std::format("group_{}", gr.name), us, "experts plus their inputs in the mixer without context");
  }
  for (int k = 0; k < fine::kContexts; ++k) {
    double us = per_mixer;
    for (const auto& [n, v] : parts) {
      if (n == std::format("context_{}", fine::context_name(k))) us += v;
    }
    add(std::format("context_{}", fine::context_name(k)), us, "context plus one mixer over every input");
  }
  for (const auto& [n, v] : parts) add("part_" + n, v, "the part alone");
  add("part_mixer_15x1", m15_1, "15 inputs, 1 mixer");
  add("part_mixer_15x4", m15_4, "15 inputs, 4 mixers");
  add("part_grain", grain_us, "grain noise");
  add("part_relock", relock_us, "relock");
  merge_csv(c.results / "g_fine_cost.csv", "effect,component,us_per_pixel,ms_per_frame,evaluated_pixels,load,note", rows);
  log(std::format("bench {}: load {:.2f} ({}), {:.0f} evaluated pixel-channels per frame; mixer 15 inputs: {:.3f} ms with 1 mixer, {:.3f} ms with 4; "
                  "grain {:.3f} ms, relock {:.3f} ms; base {:.3f} ms",
                  ename(e), load, load < 1.5 ? "quiet" : "BUSY: provisional", evaluated, m15_1 / 1000, m15_4 / 1000, grain_us / 1000, relock_us / 1000,
                  base_us / 1000));
  for (const auto& r : rows) log(r);
}
// --- probe: a generation diagnostic ------------------------------------------------------------------------------

void probe(const Ctx& c, sim::Effect e, const std::string& mixer, double tau, bool relock, int frames) {
  const rollout::Model M = load_v1(c, e);
  const fine::Mixer mix = fine::load_mixer(mixer.empty() ? released_path(c, e) : fs::path(mixer));
  const Setting s = validation_settings()[0];
  const std::vector<float> ctl(s.begin(), s.end());
  const int idx = nearest_start(M, ctl);
  const fine::GenOptions g{tau, relock, {}};
  rollout::State a = rollout::start(M, idx, kSize, ctl, 820000);
  rollout::State b = fine::start(M, mix, idx, kSize, ctl, 820000, g);
  fine::Frame f;
  const int C = M.h.channels(), N = M.h.res * M.h.res;
  for (int k = 0; k < frames; ++k) {
    rollout::step(M, a, ctl, 820000);
    fine::step(M, mix, b, ctl, 820000, g, f);
    if (k % 10 && k != frames - 1) continue;
    double cm = 0;
    for (int i = 0; i < N; ++i) cm += dbl(b.coarse[sz(i) * sz(C) + 2]);
    const auto stat = [](const std::vector<float>& v) {
      double sum = 0, mx = 0;
      for (const float x : v) {
        sum += dbl(x);
        mx = std::max(mx, dbl(x));
      }
      return std::make_pair(sum / static_cast<double>(v.size()), mx);
    };
    const auto [va, xa] = stat(a.fine_t);
    const auto [vb, xb] = stat(b.fine_t);
    log(std::format("frame {:3d}: coarse heat mean {:.4f}; v1 fine mean {:.4f} max {:.3f}; DCM fine mean {:.4f} max {:.3f}", k, cm / N, va, xa, vb, xb));
  }
}

// --- summary -------------------------------------------------------------------------------------------------------

namespace {

// CSV rows with quoted cells (the configuration descriptions hold commas).
std::vector<std::vector<std::string>> read_quoted(const fs::path& p) {
  std::vector<std::vector<std::string>> rows;
  std::ifstream in(p);
  std::string line;
  bool header = true;
  while (std::getline(in, line)) {
    if (header) {
      header = false;
      continue;
    }
    if (line.empty()) continue;
    std::vector<std::string> cells(1);
    bool quoted = false;
    for (const char ch : line) {
      if (ch == '"') {
        quoted = !quoted;
      } else if (ch == ',' && !quoted) {
        cells.emplace_back();
      } else {
        cells.back() += ch;
      }
    }
    rows.push_back(std::move(cells));
  }
  return rows;
}

double cell(const std::vector<std::string>& r, std::size_t k) { return k < r.size() && !r[k].empty() ? std::stod(r[k]) : std::nan(""); }

std::string fmt_iv(const std::vector<double>& a, const std::vector<double>& b, int prec = 3) {
  if (a.size() != b.size() || a.size() < 2) return "-";
  return iv(metrics::paired_bootstrap(a, b), prec);
}

}  // namespace

std::string summary_markdown(const fs::path& results) {
  std::ostringstream md;
  const std::vector<std::string> effects = {"fire", "smoke", "explosion"};
  // pilot
  if (fs::exists(results / "g_fine_pilot.csv")) {
    std::map<std::string, std::map<std::string, std::vector<double>>> b;  // effect -> method -> per run
    for (const auto& r : read_quoted(results / "g_fine_pilot.csv")) b[r[0]][r[1]].push_back(cell(r, 3));
    md << "### G1 pilot: the default mixer against the v1 lock as a predictor\n\nHeld-out bits per active pixel on the validation runs "
          "(salt 3), each pixel coded in a bin of s_q / 256; the v1 lock with a Laplace scale fitted per channel and heat-level bin. "
          "Differences paired over validation runs.\n\n| effect | v1 lock | default mixer (linear) | default mixer (log) | linear - v1 | log - linear |\n"
          "|---|---:|---:|---:|---|---|\n";
    for (const auto& e : effects) {
      if (!b.contains(e)) continue;
      auto& m = b[e];
      md << std::format("| {} | {:.3f} | {:.3f} | {:.3f} | {} | {} |\n", e, mean_of(m["v1_lock"]), mean_of(m["default_linear"]), mean_of(m["default_log"]),
                        fmt_iv(m["default_linear"], m["v1_lock"]), fmt_iv(m["default_log"], m["default_linear"]));
    }
  }
  // search
  if (fs::exists(results / "g_fine_search_runs.csv")) {
    std::map<std::string, std::vector<double>> nb;  // effect,family,domain,seed -> nested bits per training run
    for (const auto& r : read_quoted(results / "g_fine_search_runs.csv")) nb[std::format("{},{},{},{}", r[0], r[1], r[2], r[3])].push_back(cell(r, 6));
    std::map<std::string, std::vector<std::vector<std::string>>> sr;
    if (fs::exists(results / "g_fine_search.csv")) {
      for (const auto& r : read_quoted(results / "g_fine_search.csv")) sr[std::format("{},{},{},{}", r[0], r[1], r[2], r[3])].push_back(r);
    }
    std::map<std::string, std::vector<double>> tv;  // key,rank -> validation bits per run
    if (fs::exists(results / "g_fine_topval.csv")) {
      for (const auto& r : read_quoted(results / "g_fine_topval.csv")) tv[std::format("{},{},{},{},{}", r[0], r[1], r[2], r[3], r[4])].push_back(cell(r, 6));
    }
    md << "\n### G1 search: nested held-out bits per active pixel\n\nNested leave-one-control-bin-out search (5 k-means bins of the 48 training "
          "runs' controls), 200 configurations plus 2 refinement rounds, objective Laplace bits, cost budget +1 ms per 128 x 128 frame. "
          "`nested` is the honest held-out score of the whole search procedure (mean over training runs, each predicted by a mixer chosen and "
          "trained without its bin); `global #0` is the released configuration (chosen with every bin, so its search score is optimistic), "
          "with its bits on the validation runs.\n\n| effect | family | seed | nested bits | global #0 validation bits | cost ms | global #0 configuration |\n"
          "|---|---|---:|---:|---:|---:|---|\n";
    for (const auto& [key, v] : nb) {
      std::istringstream ks(key);
      std::string e, fam, dom, seed;
      std::getline(ks, e, ',');
      std::getline(ks, fam, ',');
      std::getline(ks, dom, ',');
      std::getline(ks, seed, ',');
      std::string conf = "-", cost = "-";
      for (const auto& r : sr[key]) {
        if (r.size() > 8 && r[4] == "global" && r[5] == "0") {
          conf = r[8];
          cost = r[7];
        }
      }
      md << std::format("| {} | {} | {} | {:.4f} | {:.4f} | {} | {} |\n", e, fam, seed, mean_of(v), mean_of(tv[key + ",0"]), cost, conf);
    }
    const auto get = [&](const std::string& e, const std::string& fam, const std::string& seed) { return nb[std::format("{},{},linear,{}", e, fam, seed)]; };
    md << "\nSeed noise of the search (fire, seeds 0, 1, 2; nested bits paired over training runs): ";
    md << std::format("1 - 0: {}; 2 - 0: {}; 2 - 1: {}.\n", fmt_iv(get("fire", "hand+macro", "1"), get("fire", "hand+macro", "0"), 4),
                      fmt_iv(get("fire", "hand+macro", "2"), get("fire", "hand+macro", "0"), 4), fmt_iv(get("fire", "hand+macro", "2"), get("fire", "hand+macro", "1"), 4));
    for (const auto& e : effects) {
      if (get(e, "none", "0").empty()) continue;
      md << std::format("\nContext families ({}, nested bits paired over training runs): hand - none {}; hand+macro - hand {}; hand+macro - none {}.\n", e,
                        fmt_iv(get(e, "hand", "0"), get(e, "none", "0"), 4), fmt_iv(get(e, "hand+macro", "0"), get(e, "hand", "0"), 4),
                        fmt_iv(get(e, "hand+macro", "0"), get(e, "none", "0"), 4));
    }
  }
  // generation calibration on validation settings
  if (fs::exists(results / "g_fine_val.csv")) {
    std::map<std::string, std::map<std::string, std::vector<double>>> v;  // effect -> candidate/tau/relock -> per setting score
    for (const auto& r : read_quoted(results / "g_fine_val.csv")) v[r[0]][std::format("{} tau {} relock {}", r[1], r[2], r[3])].push_back(cell(r, 5));
    md << "\n### G1 generation on validation settings\n\nThe calibration score of `calibrate_detail` (detail spectrum distance + |log motion "
          "ratio| + log ratios of light and cover; lower is better), mean over the 10 validation settings, single shards from the nearest start "
          "point. Candidates are the search's global top 10; pass 1 is trained on the one-step rows, pass 2 adds the own-rollout windows.\n\n"
          "| effect | generator | score | - v1, paired over settings |\n|---|---|---:|---|\n";
    for (const auto& e : effects) {
      if (!v.contains(e)) continue;
      const auto& base = v[e]["v1 tau 0.00 relock 0"];
      for (const auto& [k, s] : v[e]) md << std::format("| {} | {} | {:.4f} | {} |\n", e, k, mean_of(s), k.starts_with("v1") ? "" : fmt_iv(s, base));
    }
  }
  // the test
  if (fs::exists(results / "g_fine_test_stats.csv")) {
    std::map<std::string, std::map<std::string, std::vector<std::vector<double>>>> t;  // effect -> method -> settings -> cells
    for (const auto& r : read_quoted(results / "g_fine_test_stats.csv")) {
      std::vector<double> x;
      for (std::size_t k = 3; k < r.size(); ++k) x.push_back(cell(r, k));
      t[r[0]][r[2]].push_back(x);
    }
    md << "\n### G1 test (once): B's 10 held-out settings, new seeds\n\nSingle 6 s shards (explosions: 3 s) from the nearest start point, v1 "
          "and DCM-fine through the reference with the same stepper and seeds (only the detail layer differs), against a real 10 s run "
          "(explosions: 3 s). Means over settings; differences DCM-fine - v1 paired over settings.\n\n"
          "| effect | method | spectrum L1 | motion ratio | coverage L1 | emission L1 | mean-frame PSNR |\n|---|---|---:|---:|---:|---:|---:|\n";
    const auto col = [](const std::vector<std::vector<double>>& m, std::size_t k) {
      std::vector<double> v;
      for (const auto& x : m) v.push_back(x[k]);
      return v;
    };
    for (const auto& e : effects) {
      if (!t.contains(e)) continue;
      for (const std::string name : {"real_other_seed", "v1", "dcm_fine"}) {
        const auto& m = t[e][name];
        md << std::format("| {} | {} | {:.4f} | {:.3f} | {:.4f} | {:.4f} | {:.2f} |\n", e, name, mean_of(col(m, 1)), mean_of(col(m, 2)), mean_of(col(m, 3)),
                          mean_of(col(m, 4)), mean_of(col(m, 5)));
      }
    }
    md << "\n| effect | spectrum L1 | abs log motion ratio | coverage L1 | mean-frame PSNR | calibration score |\n|---|---|---|---|---|---|\n";
    for (const auto& e : effects) {
      if (!t.contains(e)) continue;
      const auto& a = t[e]["dcm_fine"];
      const auto& b = t[e]["v1"];
      std::vector<double> la, lb;
      for (const auto& x : a) la.push_back(std::abs(std::log(std::max(1e-3, x[2]))));
      for (const auto& x : b) lb.push_back(std::abs(std::log(std::max(1e-3, x[2]))));
      md << std::format("| {} | {} | {} | {} | {} | {} |\n", e, fmt_iv(col(a, 1), col(b, 1)), fmt_iv(la, lb), fmt_iv(col(a, 3), col(b, 3), 4),
                        fmt_iv(col(a, 5), col(b, 5), 2), fmt_iv(col(a, 0), col(b, 0)));
    }
    md << "\n### G1c: the first second\n\nCalibration score and spectrum distance of the first 30 frames against the real run (looping effects: "
          "its 10 s statistics; explosions: its own first second). `usual`: v1's start (fire: a 30-frame warm-up; smoke and explosions: stored "
          "fine fields); `cold`: from the coarse state alone, no warm-up and no stored fine fields.\n\n"
          "| effect | v1 usual | v1 cold | DCM-fine usual | DCM-fine cold | DCM cold - v1 usual (score) | DCM cold - v1 usual (spectrum) |\n"
          "|---|---:|---:|---:|---:|---|---|\n";
    for (const auto& e : effects) {
      if (!t.contains(e)) continue;
      auto& m = t[e];
      md << std::format("| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {} | {} |\n", e, mean_of(col(m["g1c_v1_usual"], 0)), mean_of(col(m["g1c_v1_cold"], 0)),
                        mean_of(col(m["g1c_dcm_usual"], 0)), mean_of(col(m["g1c_dcm_cold"], 0)), fmt_iv(col(m["g1c_dcm_cold"], 0), col(m["g1c_v1_usual"], 0)),
                        fmt_iv(col(m["g1c_dcm_cold"], 1), col(m["g1c_v1_usual"], 1)));
    }
  }
  if (fs::exists(results / "g_fine_test_track.csv")) {
    std::map<std::string, std::map<std::string, std::vector<std::vector<double>>>> t;
    for (const auto& r : read_quoted(results / "g_fine_test_track.csv")) {
      std::vector<double> x;
      for (std::size_t k = 3; k < r.size(); ++k) x.push_back(cell(r, k));
      t[r[0]][r[2]].push_back(x);
    }
    md << "\n### G1 test: tracking 8 held-out runs from their true state\n\nActive PSNR against the true run at 1, 8, 30 and 60 frames; "
          "DCM-fine - v1 paired over runs.\n\n| effect | v1 1 / 8 / 30 / 60 | DCM-fine 1 / 8 / 30 / 60 | difference at 1 | 8 | 30 | 60 |\n|---|---|---|---|---|---|---|\n";
    for (const auto& e : effects) {
      if (!t.contains(e)) continue;
      const auto& a = t[e]["dcm_fine"];
      const auto& b = t[e]["v1"];
      std::string va, vb, diffs;
      for (const std::size_t k : {std::size_t{0}, std::size_t{2}, std::size_t{4}, std::size_t{5}}) {
        std::vector<double> x, y;
        for (std::size_t r = 0; r < a.size(); ++r) {
          x.push_back(a[r][k]);
          y.push_back(b[r][k]);
        }
        va += std::format("{}{:.2f}", va.empty() ? "" : " / ", mean_of(y));
        vb += std::format("{}{:.2f}", vb.empty() ? "" : " / ", mean_of(x));
        diffs += " | " + fmt_iv(x, y, 2);
      }
      md << std::format("| {} | {} | {}{} |\n", e, va, vb, diffs);
    }
  }
  return md.str();
}

void summary(const Ctx& c) {
  const std::string md = summary_markdown(c.results);
  std::print("{}", md);
  std::ofstream(c.results / "g_fine_summary.md") << "# Study G, design G1 (DCM-fine): generated tables\n\nGenerated by `nvfx_dcm fine-summary` from "
                                                     "results/experiments/g_fine_*.csv; see docs/DCM.md.\n\n"
                                                  << md;
}

}  // namespace nfx::fine_study
