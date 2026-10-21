// The detail layer's A/B bench (docs/REPORT.md §19, results/detail/): two builds of the runtime in one process, one
// built from this tree and one from the commit before the change, each as a probe module (tools/detail_bench/probe.cpp;
// the old one from tools/detail_bench/build_old.sh). Their runners play the same models, sizes, ISAs, seeds and
// controls in lockstep, frame by frame, in alternating order (ABBA), so both see the same machine; every frame of both
// is compared bit for bit (fine fields, coarse state, picture).
//
//   nvfx_detail_bench 2d  --old OLD.so --new NEW.so --models DIR [--sizes 128,256,512] [--isas avx2,avx512]
//                         [--precision float|int8] [--rounds 12] [--frames 16] [--out CSV]
//   nvfx_detail_bench api --old OLD.so --new NEW.so --models DIR [--sizes 128] [--isas avx2,avx512] [--frames 300]
//                         [--precision default|float|int8] [--out CSV]
//   nvfx_detail_bench 3d  --old OLD.so --new NEW.so --model PATH [--threads 1,4] [--isas avx2,avx512] [--rounds 10]
//                         [--frames 6] [--out CSV]
//   nvfx_detail_bench 3d-detail --new NEW.so --model PATH [--threads 1,4] [--isas avx2,avx512] [--rounds 10] [--frames 6]
//   nvfx_detail_bench summary --in CSV --out MD   (2d, 3d and 3d-detail files; also written after each run)
//
// 2d: per frame the step's and the picture's thread CPU time, and the detail stage's (study K's profile). A round is
// `frames` frames; its ratio is the new build's median over the old's. Reported: the median of the rounds' ratios with
// a 95% bootstrap interval over rounds, per model; per size and ISA the geometric mean over models with a 95% bootstrap
// interval over models. api: nvfx_render through the C API (shards, crossfades, warm-ups), pictures compared only.
// 3d: the volume C API, one continuous rollout, time per step (on one thread its thread CPU time, on more the wall
// time). 3d-detail: the new build alone, rollout3d's reference detail_step against the volume runtime's on two copies of
// one rollout (likewise timed).
#include "args.hpp"

#include <dlfcn.h>
#include <time.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <map>
#include <print>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

// --- the probes ------------------------------------------------------------------------------------------------------

struct Probe {
  void* h = nullptr;
  void* (*effect)(const char*) = nullptr;
  void (*effect_free)(void*) = nullptr;
  void* (*runner)(void*, int, int, int) = nullptr;
  void (*runner_free)(void*) = nullptr;
  void (*start)(void*, int, const float*, int, std::uint64_t) = nullptr;
  void (*step)(void*, const float*, int, std::uint64_t) = nullptr;
  const std::uint8_t* (*render)(void*) = nullptr;
  const float* (*field)(void*, int, std::size_t*) = nullptr;
  void (*stages)(void*, double*) = nullptr;
  void* (*api_effect)(const char*) = nullptr;
  void (*api_effect_free)(void*) = nullptr;
  void* (*api_instance)(void*, int, int, int, const float*, int, std::uint64_t) = nullptr;
  void (*api_instance_free)(void*) = nullptr;
  int (*api_render)(void*, double, std::uint8_t*, std::size_t) = nullptr;
  void* (*volume_effect)(const char*) = nullptr;
  void (*volume_effect_free)(void*) = nullptr;
  void* (*volume)(void*, int, int, int, const float*, int, std::uint64_t, int*) = nullptr;
  void (*volume_free)(void*) = nullptr;
  int (*volume_step_to)(void*, double) = nullptr;
  int (*volume_read)(void*, int, float*, std::size_t) = nullptr;
  std::size_t (*volume_bytes)(void*) = nullptr;
  void* (*d3_pair)(const char*, int, int, int, const float*, int, std::uint64_t) = nullptr;  // new builds only
  void (*d3_pair_free)(void*) = nullptr;
  long (*d3_pair_step)(void*, int, double*) = nullptr;
};

template <class F>
void bind(void* h, const char* name, F& f, bool required = true) {
  f = reinterpret_cast<F>(dlsym(h, name));
  if (!f && required) throw std::runtime_error(std::format("probe: no {}", name));
}

Probe open_probe(const std::string& path) {
  Probe p;
  // a copy under its own name, so two probes of the same file name load as two libraries
  p.h = dlopen(fs::absolute(path).c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!p.h) throw std::runtime_error(std::format("cannot load {}: {}", path, dlerror()));
  bind(p.h, "probe_effect", p.effect);
  bind(p.h, "probe_effect_free", p.effect_free);
  bind(p.h, "probe_runner", p.runner);
  bind(p.h, "probe_runner_free", p.runner_free);
  bind(p.h, "probe_start", p.start);
  bind(p.h, "probe_step", p.step);
  bind(p.h, "probe_render", p.render);
  bind(p.h, "probe_field", p.field);
  bind(p.h, "probe_stages", p.stages);
  bind(p.h, "probe_api_effect", p.api_effect);
  bind(p.h, "probe_api_effect_free", p.api_effect_free);
  bind(p.h, "probe_api_instance", p.api_instance);
  bind(p.h, "probe_api_instance_free", p.api_instance_free);
  bind(p.h, "probe_api_render", p.api_render);
  bind(p.h, "probe_volume_effect", p.volume_effect);
  bind(p.h, "probe_volume_effect_free", p.volume_effect_free);
  bind(p.h, "probe_volume", p.volume);
  bind(p.h, "probe_volume_free", p.volume_free);
  bind(p.h, "probe_volume_step_to", p.volume_step_to);
  bind(p.h, "probe_volume_read", p.volume_read);
  bind(p.h, "probe_volume_scratch_bytes", p.volume_bytes);
  bind(p.h, "probe_d3_pair", p.d3_pair, false);
  bind(p.h, "probe_d3_pair_free", p.d3_pair_free, false);
  bind(p.h, "probe_d3_pair_step", p.d3_pair_step, false);
  return p;
}

// --- statistics -------------------------------------------------------------------------------------------------------

double median(std::vector<double> v) {
  if (v.empty()) return std::nan("");
  std::ranges::sort(v);
  const std::size_t n = v.size();
  return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}
double geomean(const std::vector<double>& v) {
  double s = 0;
  for (const double x : v) s += std::log(x);
  return std::exp(s / static_cast<double>(v.size()));
}
// A 95% percentile bootstrap interval of stat over resamples of v.
template <class F>
std::array<double, 2> bootstrap(const std::vector<double>& v, F stat, int B = 4000) {
  std::mt19937_64 rng(20261021);
  std::uniform_int_distribution<std::size_t> pick(0, v.size() - 1);
  std::vector<double> s(static_cast<std::size_t>(B)), r(v.size());
  for (auto& x : s) {
    for (auto& y : r) y = v[pick(rng)];
    x = stat(r);
  }
  std::ranges::sort(s);
  return {s[static_cast<std::size_t>(0.025 * B)], s[static_cast<std::size_t>(0.975 * B) - 1]};
}

double thread_ms() {
  timespec t{};
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
  return static_cast<double>(t.tv_sec) * 1e3 + static_cast<double>(t.tv_nsec) * 1e-6;
}
double wall_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  for (std::string w; std::getline(ss, w, ',');) {
    if (!w.empty()) out.push_back(w);
  }
  return out;
}
int isa_of(const std::string& s) {
  if (s == "avx512") return 3;
  if (s == "avx2") return 2;
  if (s == "baseline") return 1;
  if (s == "auto") return 0;
  throw std::invalid_argument("isa: avx2, avx512, baseline or auto");
}

std::vector<fs::path> models_in(const fs::path& dir, const std::string& names) {
  std::vector<fs::path> v;
  const auto only = split(names);
  for (const auto& e : fs::directory_iterator(dir)) {
    if (e.path().extension() != ".nvfx") continue;
    if (!only.empty() && std::ranges::find(only, e.path().stem().string()) == only.end()) continue;
    v.push_back(e.path());
  }
  std::ranges::sort(v);
  if (v.empty()) throw std::runtime_error("no models in " + dir.string());
  return v;
}

// The rollout's play length before a restart: a one-shot (not looping, h.frames frames) restarts 12 frames before its
// end, as study K's timing stops there; a loop plays on.
int play_length(const fs::path& model) {
  std::ifstream in(model, std::ios::binary);
  char head[96] = {};
  in.read(head, sizeof head);
  std::int32_t frames = 0;
  std::memcpy(&frames, head + 12 + 6 * 4, 4);  // magic, version, then res, hidden, memory, jacobi, n_controls, n_age, frames
  const bool loop = head[12 + 10 * 4 + 32 + 4] != 0;  // after the ten integers, the name (32 bytes) and fps
  return loop || frames <= 24 ? 1 << 30 : frames - 12;
}

bool same(const float* a, const float* b, std::size_t n) { return std::memcmp(a, b, n * 4) == 0; }

// --- 2D ---------------------------------------------------------------------------------------------------------------

constexpr float kCtl[3] = {0.6f, 0.5f, 0.6f};

void run_2d(const nfx::tools::Args& a) {
  const Probe po = open_probe(a.need("old")), pn = open_probe(a.need("new"));
  const auto models = models_in(a.need("models"), a.str("names"));
  const auto sizes = split(a.str("sizes", "128,256,512"));
  const auto isas = split(a.str("isas", "avx2,avx512"));
  const int int8 = a.str("precision", "float") == "int8" ? 1 : 0;
  const int rounds = a.i("rounds", 12), frames = a.i("frames", 16), warm = a.i("warmup", 8);
  const std::uint64_t seed = a.u64("seed", 1);
  const fs::path out = a.str("out", "results/detail/frames_2d.csv");
  if (out.has_parent_path()) fs::create_directories(out.parent_path());
  std::ofstream csv(out);
  csv << "model,size,isa,precision,round,frames,old_step_ms,new_step_ms,old_frame_ms,new_frame_ms,old_detail_ms,new_detail_ms,differing_frames,min_psnr_db\n";
  long total_frames = 0, total_bad = 0;
  for (const auto& path : models) {
    void* eo = po.effect(path.c_str());
    void* en = pn.effect(path.c_str());
    if (!eo || !en) throw std::runtime_error("cannot load " + path.string());
    const int length = play_length(path);
    for (const auto& ss : sizes) {
      const int size = std::stoi(ss);
      for (const auto& is : isas) {
        const int isa = isa_of(is);
        void* ro = po.runner(eo, size, isa, int8);
        void* rn = pn.runner(en, size, isa, int8);
        if (!ro || !rn) throw std::runtime_error("cannot make runners");
        const auto begin = [&] {
          po.start(ro, 0, kCtl, 3, seed);
          pn.start(rn, 0, kCtl, 3, seed);
          for (int f = 0; f < warm; ++f) {
            po.step(ro, kCtl, 3, seed);
            pn.step(rn, kCtl, 3, seed);
          }
        };
        begin();
        int played = warm;
        std::string line;
        std::vector<double> ratios;
        long bad_frames = 0;
        for (int r = 0; r < rounds; ++r) {
          std::vector<double> st[2], fr[2], de[2];
          int bad = 0;
          double worst = std::numeric_limits<double>::infinity();  // the round's lowest picture PSNR, old against new
          for (int f = 0; f < frames; ++f) {
            if (played >= length) {
              begin();
              played = warm;
            }
            for (int k = 0; k < 2; ++k) {
              const bool newer = ((r + f + k) % 2) != 0;  // ABBA
              const Probe& p = newer ? pn : po;
              void* run = newer ? rn : ro;
              double s0[6], s1[6];
              p.stages(run, s0);
              const double t0 = thread_ms();
              p.step(run, kCtl, 3, seed);
              const double t1 = thread_ms();
              p.render(run);
              const double t2 = thread_ms();
              p.stages(run, s1);
              st[newer].push_back(t1 - t0);
              fr[newer].push_back(t2 - t0);
              de[newer].push_back(s1[3] - s0[3]);
            }
            ++played;
            bool ok = true;
            for (int w = 0; w < 4; ++w) {
              std::size_t n0 = 0, n1 = 0;
              const float* x = po.field(ro, w, &n0);
              const float* y = pn.field(rn, w, &n1);
              ok = ok && n0 == n1 && same(x, y, n0);
            }
            const std::size_t bytes = static_cast<std::size_t>(size) * static_cast<std::size_t>(size) * 4;
            const std::uint8_t* pa = po.render(ro);
            const std::uint8_t* pb = pn.render(rn);
            if (std::memcmp(pa, pb, bytes) != 0) {
              ok = false;
              double se = 0;
              for (std::size_t q = 0; q < bytes; ++q) se += (static_cast<double>(pa[q]) - pb[q]) * (static_cast<double>(pa[q]) - pb[q]);
              worst = std::min(worst, 10.0 * std::log10(255.0 * 255.0 / (se / static_cast<double>(bytes))));
            }
            bad += !ok;
          }
          bad_frames += bad;
          ratios.push_back(median(fr[1]) / median(fr[0]));
          csv << std::format("{},{},{},{},{},{},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{},{:.2f}\n", path.stem().string(), size, is, int8 ? "int8" : "float", r,
                             frames, median(st[0]), median(st[1]), median(fr[0]), median(fr[1]), median(de[0]), median(de[1]), bad, worst);
        }
        csv.flush();
        total_frames += static_cast<long>(rounds) * frames;
        total_bad += bad_frames;
        std::println("{:>10} {:>3} {:>6}: frame new/old {:.3f}  differing frames {}", path.stem().string(), size, is, median(ratios), bad_frames);
        po.runner_free(ro);
        pn.runner_free(rn);
      }
    }
    po.effect_free(eo);
    pn.effect_free(en);
  }
  std::println("{} frames compared, {} differ", total_frames, total_bad);
}

// --- the C API: pictures through instances ----------------------------------------------------------------------------

void run_api(const nfx::tools::Args& a) {
  const Probe po = open_probe(a.need("old")), pn = open_probe(a.need("new"));
  const auto models = models_in(a.need("models"), a.str("names"));
  const auto sizes = split(a.str("sizes", "128"));
  const auto isas = split(a.str("isas", "avx2,avx512"));
  const std::string pr = a.str("precision", "default");
  const int prec = pr == "float" ? 1 : pr == "int8" ? 2 : 0;
  const int frames = a.i("frames", 300);
  const std::uint64_t seed = a.u64("seed", 7);
  const fs::path out = a.str("out", "results/detail/identity_api.csv");
  if (out.has_parent_path()) fs::create_directories(out.parent_path());
  std::ofstream csv(out);
  csv << "model,size,isa,precision,frames,differing_frames\n";
  long total = 0, total_bad = 0;
  for (const auto& path : models) {
    void* eo = po.api_effect(path.c_str());
    void* en = pn.api_effect(path.c_str());
    if (!eo || !en) throw std::runtime_error("cannot load " + path.string());
    for (const auto& ss : sizes) {
      const int size = std::stoi(ss);
      for (const auto& is : isas) {
        void* io = po.api_instance(eo, size, isa_of(is), prec, kCtl, 3, seed);
        void* in = pn.api_instance(en, size, isa_of(is), prec, kCtl, 3, seed);
        if (!io || !in) throw std::runtime_error("cannot make instances");
        const std::size_t bytes = static_cast<std::size_t>(size) * static_cast<std::size_t>(size) * 4;
        std::vector<std::uint8_t> x(bytes), y(bytes);
        int bad = 0;
        for (int f = 0; f < frames; ++f) {
          const double t = f / 30.0;
          if (po.api_render(io, t, x.data(), static_cast<std::size_t>(size) * 4) || pn.api_render(in, t, y.data(), static_cast<std::size_t>(size) * 4)) {
            throw std::runtime_error("render failed");
          }
          bad += x != y;
        }
        csv << std::format("{},{},{},{},{},{}\n", path.stem().string(), size, is, pr, frames, bad);
        total += frames;
        total_bad += bad;
        std::println("{:>10} {:>3} {:>6} {}: {} frames, {} differ", path.stem().string(), size, is, pr, frames, bad);
        po.api_instance_free(io);
        pn.api_instance_free(in);
      }
    }
    po.api_effect_free(eo);
    pn.api_effect_free(en);
  }
  std::println("{} pictures compared, {} differ", total, total_bad);
}

// --- 3D ---------------------------------------------------------------------------------------------------------------

void run_3d(const nfx::tools::Args& a) {
  const Probe po = open_probe(a.need("old")), pn = open_probe(a.need("new"));
  const std::string model = a.need("model");
  const auto threads = split(a.str("threads", "1,4"));
  const auto isas = split(a.str("isas", "avx2,avx512"));
  const int rounds = a.i("rounds", 10), frames = a.i("frames", 6);
  const std::uint64_t seed = a.u64("seed", 11);
  const float c[3] = {0.7f, 0.5f, 0.6f};
  const fs::path out = a.str("out", "results/detail/volume_3d.csv");
  if (out.has_parent_path()) fs::create_directories(out.parent_path());
  std::ofstream csv(out);
  csv << "model,threads,isa,round,frames,old_step_ms,new_step_ms,differing_frames,old_bytes,new_bytes\n";
  void* eo = po.volume_effect(model.c_str());
  void* en = pn.volume_effect(model.c_str());
  if (!eo || !en) throw std::runtime_error("cannot load " + model);
  for (const auto& ts : threads) {
    for (const auto& is : isas) {
      int size[3] = {};
      const bool one = std::stoi(ts) == 1;  // one thread: its CPU time; more: wall time
      void* vo = po.volume(eo, std::stoi(ts), isa_of(is), 0, c, 3, seed, size);
      void* vn = pn.volume(en, std::stoi(ts), isa_of(is), 0, c, 3, seed, size);
      if (!vo || !vn) throw std::runtime_error("cannot make volumes");
      const std::size_t n = static_cast<std::size_t>(size[0]) * static_cast<std::size_t>(size[1]) * static_cast<std::size_t>(size[2]);
      std::vector<float> x(n), y(n);
      // frame 0 (the start and its warm-up), not timed
      if (po.volume_step_to(vo, 0.5 / 30.0) || pn.volume_step_to(vn, 0.5 / 30.0)) throw std::runtime_error("step failed");
      long frame = 0, bad_total = 0;
      for (int r = 0; r < rounds; ++r) {
        std::vector<double> t[2];
        int bad = 0;
        for (int f = 0; f < frames; ++f) {
          ++frame;
          const double at = (static_cast<double>(frame) + 0.5) / 30.0;  // the models play at 30 fps
          for (int k = 0; k < 2; ++k) {
            const bool newer = ((r + f + k) % 2) != 0;
            const double t0 = one ? thread_ms() : wall_ms();
            if ((newer ? pn : po).volume_step_to(newer ? vn : vo, at)) throw std::runtime_error("step failed");
            t[newer].push_back((one ? thread_ms() : wall_ms()) - t0);
          }
          bool ok = true;
          for (int field = 0; field < 2; ++field) {
            po.volume_read(vo, field, x.data(), n);
            pn.volume_read(vn, field, y.data(), n);
            ok = ok && same(x.data(), y.data(), n);
          }
          bad += !ok;
        }
        bad_total += bad;
        csv << std::format("{},{},{},{},{},{:.4f},{:.4f},{},{},{}\n", fs::path(model).stem().string(), ts, is, r, frames, median(t[0]), median(t[1]), bad,
                           po.volume_bytes(vo), pn.volume_bytes(vn));
        csv.flush();
        std::println("3d {} threads {:>6} round {:>2}: step old {:.2f} new {:.2f} ms, differing frames {}", ts, is, r, median(t[0]), median(t[1]), bad);
      }
      std::println("3d {} threads {}: {} frames compared, {} differ", ts, is, frame, bad_total);
      po.volume_free(vo);
      pn.volume_free(vn);
    }
  }
  po.volume_effect_free(eo);
  pn.volume_effect_free(en);
}

void run_3d_detail(const nfx::tools::Args& a) {
  const Probe pn = open_probe(a.need("new"));
  if (!pn.d3_pair) throw std::runtime_error("the probe has no 3D detail pair (a build before the volume runtime's detail layer)");
  const std::string model = a.need("model");
  const auto threads = split(a.str("threads", "1,4"));
  const auto isas = split(a.str("isas", "avx2,avx512"));
  const int rounds = a.i("rounds", 10), frames = a.i("frames", 6);
  const float c[3] = {0.7f, 0.5f, 0.6f};
  const fs::path out = a.str("out", "results/detail/detail_3d.csv");
  if (out.has_parent_path()) fs::create_directories(out.parent_path());
  std::ofstream csv(out);
  csv << "model,threads,isa,round,frames,reference_ms,runtime_ms,differing_cells\n";
  for (const auto& ts : threads) {
    for (const auto& is : isas) {
      void* p = pn.d3_pair(model.c_str(), isa_of(is), std::stoi(ts), 0, c, 3, a.u64("seed", 11));
      if (!p) throw std::runtime_error("cannot make the pair");
      for (int r = 0; r < rounds; ++r) {
        std::vector<double> t[2];
        long bad = 0;
        for (int f = 0; f < frames; ++f) {
          double ms[2];
          bad += pn.d3_pair_step(p, (r + f) % 2, ms);
          t[0].push_back(ms[0]);
          t[1].push_back(ms[1]);
        }
        csv << std::format("{},{},{},{},{},{:.4f},{:.4f},{}\n", fs::path(model).stem().string(), ts, is, r, frames, median(t[0]), median(t[1]), bad);
        csv.flush();
        std::println("3d detail {} threads {:>6} round {:>2}: reference {:.2f} runtime {:.2f} ms, differing cells {}", ts, is, r, median(t[0]), median(t[1]), bad);
      }
      pn.d3_pair_free(p);
    }
  }
}

// --- summary ------------------------------------------------------------------------------------------------------------

struct Table {
  std::vector<std::string> head;
  std::vector<std::vector<std::string>> rows;
};
Table read_csv(const fs::path& p) {
  std::ifstream in(p);
  if (!in) throw std::runtime_error("cannot read " + p.string());
  Table t;
  std::string line;
  std::getline(in, line);
  t.head = split(line);
  while (std::getline(in, line)) {
    if (!line.empty()) t.rows.push_back(split(line));
  }
  return t;
}
std::size_t col(const Table& t, const std::string& name) {
  const auto it = std::ranges::find(t.head, name);
  if (it == t.head.end()) throw std::runtime_error("no column " + name);
  return static_cast<std::size_t>(it - t.head.begin());
}
std::string ci(double m, std::array<double, 2> c, int digits = 3) { return std::format("{:.{}f} [{:.{}f}, {:.{}f}]", m, digits, c[0], digits, c[1], digits); }

void summary(const nfx::tools::Args& a) {
  const fs::path in = a.need("in");
  const Table t = read_csv(in);
  std::ostringstream md;
  const auto med = [](const std::vector<double>& v) { return median(v); };
  if (std::ranges::find(t.head, "old_frame_ms") != t.head.end()) {  // 2d
    const std::size_t cm = col(t, "model"), cs = col(t, "size"), ci_ = col(t, "isa"), cos = col(t, "old_step_ms"), cns = col(t, "new_step_ms"), cof = col(t, "old_frame_ms"),
                      cnf = col(t, "new_frame_ms"), cod = col(t, "old_detail_ms"), cnd = col(t, "new_detail_ms"), cb = col(t, "differing_frames"), cf = col(t, "frames");
    struct Cell {
      std::vector<double> os, ns, of, nf, od, nd, rs, rf, rd;
      long frames = 0, bad = 0;
    };
    std::map<std::tuple<int, std::string, std::string>, Cell> cells;  // size, isa, model
    for (const auto& r : t.rows) {
      Cell& c = cells[{std::stoi(r[cs]), r[ci_], r[cm]}];
      const double os = std::stod(r[cos]), ns = std::stod(r[cns]), of = std::stod(r[cof]), nf = std::stod(r[cnf]), od = std::stod(r[cod]), nd = std::stod(r[cnd]);
      c.os.push_back(os);
      c.ns.push_back(ns);
      c.of.push_back(of);
      c.nf.push_back(nf);
      c.od.push_back(od);
      c.nd.push_back(nd);
      c.rs.push_back(ns / os);
      c.rf.push_back(nf / of);
      c.rd.push_back(nd / od);
      c.frames += std::stol(r[cf]);
      c.bad += std::stol(r[cb]);
    }
    md << "## 2D: new / old, per size and ISA (geometric mean over models of each model's median ratio over rounds, 95% bootstrap over models)\n\n";
    md << "| size | ISA | models | detail | step | frame | old detail ms | new detail ms | old frame ms | new frame ms | frames compared | differing |\n";
    md << "|---:|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|\n";
    std::map<std::pair<int, std::string>, std::vector<const Cell*>> groups;
    for (const auto& [k, c] : cells) groups[{std::get<0>(k), std::get<1>(k)}].push_back(&c);
    for (const auto& [k, cs2] : groups) {
      std::vector<double> rd, rs, rf, od, nd, of, nf;
      long frames = 0, bad = 0;
      for (const Cell* c : cs2) {
        rd.push_back(median(c->rd));
        rs.push_back(median(c->rs));
        rf.push_back(median(c->rf));
        od.push_back(median(c->od));
        nd.push_back(median(c->nd));
        of.push_back(median(c->of));
        nf.push_back(median(c->nf));
        frames += c->frames;
        bad += c->bad;
      }
      const auto gm = [](const std::vector<double>& v) { return geomean(v); };
      md << std::format("| {} | {} | {} | {} | {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {} | {} |\n", k.first, k.second, cs2.size(), ci(gm(rd), bootstrap(rd, gm)),
                        ci(gm(rs), bootstrap(rs, gm)), ci(gm(rf), bootstrap(rf, gm)), median(od), median(nd), median(of), median(nf), frames, bad);
    }
    md << "\n(old and new ms: medians over models of each model's median; thread CPU time.)\n\n";
    md << "## 2D per model: new / old, median over rounds [95% bootstrap over rounds]\n\n";
    md << "| model | size | ISA | detail | step | frame | old detail ms | new detail ms | old frame ms | new frame ms | differing frames |\n";
    md << "|---|---:|---|---|---|---|---:|---:|---:|---:|---:|\n";
    for (const auto& [k, c] : cells) {
      md << std::format("| {} | {} | {} | {} | {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {} |\n", std::get<2>(k), std::get<0>(k), std::get<1>(k), ci(median(c.rd), bootstrap(c.rd, med)),
                        ci(median(c.rs), bootstrap(c.rs, med)), ci(median(c.rf), bootstrap(c.rf, med)), median(c.od), median(c.nd), median(c.of), median(c.nf), c.bad);
    }
  } else if (std::ranges::find(t.head, "old_step_ms") != t.head.end()) {  // 3d
    const std::size_t ct = col(t, "threads"), ci_ = col(t, "isa"), co = col(t, "old_step_ms"), cn = col(t, "new_step_ms"), cb = col(t, "differing_frames"), cf = col(t, "frames");
    std::map<std::pair<int, std::string>, std::array<std::vector<double>, 3>> g;
    std::map<std::pair<int, std::string>, std::array<long, 2>> n;
    for (const auto& r : t.rows) {
      auto& v = g[{std::stoi(r[ct]), r[ci_]}];
      v[0].push_back(std::stod(r[co]));
      v[1].push_back(std::stod(r[cn]));
      v[2].push_back(std::stod(r[cn]) / std::stod(r[co]));
      auto& m = n[{std::stoi(r[ct]), r[ci_]}];
      m[0] += std::stol(r[cf]);
      m[1] += std::stol(r[cb]);
    }
    md << "## 3D volume step: new / old (median over rounds [95% bootstrap over rounds]; thread CPU time on 1 thread, wall time on more)\n\n";
    md << "| threads | ISA | old step ms | new step ms | new / old | frames compared | differing |\n|---:|---|---|---|---|---:|---:|\n";
    for (const auto& [k, v] : g) {
      md << std::format("| {} | {} | {} | {} | {} | {} | {} |\n", k.first, k.second, ci(median(v[0]), bootstrap(v[0], med), 2), ci(median(v[1]), bootstrap(v[1], med), 2),
                        ci(median(v[2]), bootstrap(v[2], med)), n[k][0], n[k][1]);
    }
  } else {  // 3d-detail
    const std::size_t ct = col(t, "threads"), ci_ = col(t, "isa"), co = col(t, "reference_ms"), cn = col(t, "runtime_ms"), cb = col(t, "differing_cells"), cf = col(t, "frames");
    std::map<std::pair<int, std::string>, std::array<std::vector<double>, 3>> g;
    std::map<std::pair<int, std::string>, std::array<long, 2>> n;
    for (const auto& r : t.rows) {
      auto& v = g[{std::stoi(r[ct]), r[ci_]}];
      v[0].push_back(std::stod(r[co]));
      v[1].push_back(std::stod(r[cn]));
      v[2].push_back(std::stod(r[cn]) / std::stod(r[co]));
      auto& m = n[{std::stoi(r[ct]), r[ci_]}];
      m[0] += std::stol(r[cf]);
      m[1] += std::stol(r[cb]);
    }
    md << "## 3D detail layer alone: the volume runtime's / rollout3d::detail_step (median over rounds [95% bootstrap]; thread CPU time on 1 thread, wall time on more)\n\n";
    md << "| threads | ISA | reference ms | runtime ms | runtime / reference | steps compared | differing cells |\n|---:|---|---|---|---|---:|---:|\n";
    for (const auto& [k, v] : g) {
      md << std::format("| {} | {} | {} | {} | {} | {} | {} |\n", k.first, k.second, ci(median(v[0]), bootstrap(v[0], med), 2), ci(median(v[1]), bootstrap(v[1], med), 2),
                        ci(median(v[2]), bootstrap(v[2], med)), n[k][0], n[k][1]);
    }
  }
  const std::string text = md.str();
  if (a.has("out")) {
    std::ofstream(a.str("out")) << text;
  }
  std::print("{}", text);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    nfx::tools::Args a(argc, argv, {});
    const auto& pos = a.positional();
    const std::string cmd = pos.empty() ? "" : pos[0];
    if (cmd == "2d") run_2d(a);
    else if (cmd == "api") run_api(a);
    else if (cmd == "3d") run_3d(a);
    else if (cmd == "3d-detail") run_3d_detail(a);
    else if (cmd == "summary") summary(a);
    else {
      std::println(stderr, "usage: nvfx_detail_bench 2d|api|3d|3d-detail|summary ... (see the top of tools/nvfx_detail_bench.cpp)");
      return 2;
    }
    a.warn_unused();
  } catch (const std::exception& e) {
    std::println(stderr, "nvfx_detail_bench: {}", e.what());
    return 1;
  }
  return 0;
}
