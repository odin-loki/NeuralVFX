// nvfx_study_h: the measurements of study H (docs/DCM.md §9) that are not the coder's (nvfx_pack --h3 has those).
//
//   nvfx_study_h --h2 [--models DIR] [--reps N] [--frames N] [--isa avx2|avx512|baseline] [--only EFFECT] [--out CSV]
//       H2, run-aware fields: the runtime's rollout step and learned renderer with skipping on and off, on the study D
//       effects at the fireball's sizes. Two runners in lock step from the same start; every frame each is stepped
//       and rendered, timed by thread CPU time, alternating which goes first, and their fields and pictures are
//       compared to the bit. The least of N repetitions per frame; sums over the frames.
//   nvfx_study_h --h1 [--models DIR] [--reps N] [--out CSV] FILES...
//       H1, computing on LZ78- and grammar-compressed data. For each frame model (.nvfx) the per-frame blend of its
//       feature volume (slice = sum over bases and the two time slices of weight * dequantised plane, as the runtime's
//       blend_slice) dense (AVX2, as the runtime) and on compressed forms of the planes' non-zero (column, value)
//       pairs (zeros add nothing, so they are left out as in a sparse format): LZ78 (a phrase is its parent plus one
//       pair; a phrase's weight is pushed to its parent, so the cost follows the dictionary), RePair (pairs of symbols
//       replaced by rules until no pair repeats; weights pushed down the rules) and, to tell the grammar's part from
//       the sparse format's, plain sparse rows (CSR). Also the rollout effects' stored fine fields (blends of start
//       points) and weight tables (W x per
//       pixel, partial sums built up the dictionary), and synthetic blends of fields that are empty but for a blob
//       covering 50% down to 2% (where the crossover lies). Sizes of the compressed forms, times, and the largest
//       difference from the dense result.
//
// Timings are thread CPU time on one core (pin it with taskset); the machine may be shared, so least of N.
#include "args.hpp"
#include "compose.hpp"

#include <neuralfx/model.hpp>
#include <neuralfx/rollout.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <format>
#include <fstream>
#include <print>
#include <random>
#include <stdfloat>
#include <unordered_map>

namespace fs = std::filesystem;
using namespace nfx;

namespace {

std::size_t zs(int v) { return static_cast<std::size_t>(v); }

double cpu_now() {
  timespec t{};
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
  return static_cast<double>(t.tv_sec) + 1e-9 * static_cast<double>(t.tv_nsec);
}

template <class F>
double timed(F&& f) {
  const double t0 = cpu_now();
  f();
  return cpu_now() - t0;
}

// --- H2 ----------------------------------------------------------------------------------------------------------------

rt::RolloutEffect load_effect(const fs::path& p) {
  auto m = rollout::load_model(p);
  if (!m) throw std::runtime_error(std::format("{}: {}", p.string(), m.error()));
  rt::RolloutEffect e;
  e.m = std::move(*m);
  return e;
}

// The start point with the most heat (the fireball starts its explosion from the strongest).
int strongest(const rollout::Model& m) {
  int best = 0;
  double most = -1;
  for (std::size_t k = 0; k < m.starts.size(); ++k) {
    double s = 0;
    for (std::size_t i = 2; i < m.starts[k].coarse.size(); i += rollout::kPhys) s += m.starts[k].coarse[i];
    if (s > most) {
      most = s;
      best = static_cast<int>(k);
    }
  }
  return best;
}

int h2(const tools::Args& a) {
  const fs::path dir = a.str("models", "/root/nvfx-data/experiments/models/d");
  const int reps = a.i("reps", 5), frames = a.i("frames", 90);
  const std::string isa_s = a.str("isa", "avx2");
  const compose::Isa isa = isa_s == "baseline" ? compose::Isa::base : isa_s == "avx512" ? compose::Isa::avx512 : compose::Isa::avx2;
  std::ofstream csv(a.str("out", "results/experiments/h2_step.csv"));
  csv << "effect,size,start,isa,frame,not_zero,step_ms_skip,step_ms_full,render_ms_skip,render_ms_full\n";
  std::println("| effect | size | not +0 (mean) | step ms, skip | step ms, all | ratio | render ms, skip | render ms, all | ratio |");
  std::println("|---|---:|---:|---:|---:|---:|---:|---:|---:|");
  struct Case {
    std::string file;
    int size;
    bool strongest;
  };
  // The fireball's sizes at 1280 x 720: main tiles 384, the wreck's explosion 256, the wreck's fire 192, fires 128.
  const std::vector<Case> cases = {{"explosion", 384, true}, {"explosion", 256, true}, {"smoke", 384, false}, {"fire", 192, false}, {"fire", 128, false}};
  const std::string only = a.str("only", "");
  for (const Case& c : cases) {
    if (!only.empty() && c.file != only) continue;
    const rt::RolloutEffect e = load_effect(dir / (c.file + ".nvfx"));
    const int start = c.strongest ? strongest(e.m) : 0;
    const auto& controls = e.m.starts[zs(start)].controls;
    std::vector<double> ss(zs(frames), 1e9), sf(zs(frames), 1e9), rs(zs(frames), 1e9), rf(zs(frames), 1e9), nz(zs(frames), 0);
    for (int r = 0; r < reps; ++r) {
      // created in alternating order, so that neither runner always gets the same place in memory (cache aliasing
      // between two instances can differ by several percent for the same work)
      std::unique_ptr<rt::RolloutRunner> skip, full;
      if (r % 2 == 0) {
        skip = compose::make_runner(e, c.size, isa);
        full = compose::make_runner(e, c.size, isa);
      } else {
        full = compose::make_runner(e, c.size, isa);
        skip = compose::make_runner(e, c.size, isa);
      }
      full->skip_empty(false);
      skip->start(start, controls, 11);
      full->start(start, controls, 11);
      std::vector<std::uint8_t> pa(zs(c.size) * zs(c.size) * 4), pb(pa.size());
      for (int f = 0; f < frames; ++f) {
        const bool skip_first = ((f + r) & 1) == 0;
        double a1 = 0, b1 = 0, a2 = 0, b2 = 0;
        if (skip_first) {
          a1 = timed([&] { skip->step(controls, 11); });
          b1 = timed([&] { full->step(controls, 11); });
          a2 = timed([&] { skip->render(rt::FrameInput{}, pa.data(), zs(c.size) * 4); });
          b2 = timed([&] { full->render(rt::FrameInput{}, pb.data(), zs(c.size) * 4); });
        } else {
          b1 = timed([&] { full->step(controls, 11); });
          a1 = timed([&] { skip->step(controls, 11); });
          b2 = timed([&] { full->render(rt::FrameInput{}, pb.data(), zs(c.size) * 4); });
          a2 = timed([&] { skip->render(rt::FrameInput{}, pa.data(), zs(c.size) * 4); });
        }
        const auto ft = skip->fine_heat(), fd = skip->fine_soot();
        if (std::memcmp(ft.data(), full->fine_heat().data(), ft.size() * 4) != 0 || std::memcmp(fd.data(), full->fine_soot().data(), fd.size() * 4) != 0 ||
            pa != pb) {
          throw std::runtime_error(std::format("{} {}: skipping changed frame {}", c.file, c.size, f));
        }
        std::size_t n = 0;
        for (std::size_t i = 0; i < ft.size(); ++i) n += ft[i] != 0.f || fd[i] != 0.f;
        nz[zs(f)] = static_cast<double>(n) / static_cast<double>(ft.size());
        ss[zs(f)] = std::min(ss[zs(f)], a1);
        sf[zs(f)] = std::min(sf[zs(f)], b1);
        rs[zs(f)] = std::min(rs[zs(f)], a2);
        rf[zs(f)] = std::min(rf[zs(f)], b2);
      }
    }
    double s1 = 0, s2 = 0, s3 = 0, s4 = 0, m = 0;
    for (int f = 0; f < frames; ++f) {
      csv << std::format("{},{},{},{},{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n", c.file, c.size, start, isa_s, f, nz[zs(f)], 1e3 * ss[zs(f)], 1e3 * sf[zs(f)],
                         1e3 * rs[zs(f)], 1e3 * rf[zs(f)]);
      s1 += ss[zs(f)];
      s2 += sf[zs(f)];
      s3 += rs[zs(f)];
      s4 += rf[zs(f)];
      m += nz[zs(f)];
    }
    const double n = static_cast<double>(frames);
    std::println("| {} | {} | {:.2f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |", c.file, c.size, m / n, 1e3 * s1 / n, 1e3 * s2 / n, s1 / s2,
                 1e3 * s3 / n, 1e3 * s4 / n, s3 / s4);
    csv.flush();
  }
  return 0;
}

// --- H1 ----------------------------------------------------------------------------------------------------------------

// A dictionary of LZ78 phrases over the rows' non-zero (column, value) pairs, in column order (zeros contribute nothing
// to a product and are left out, as in a sparse format): node k is its parent's phrase (or nothing, parent -1)
// followed by value val[k] at column col[k]. A row is a list of phrases that tile its pairs.
struct Lz78 {
  std::vector<std::int32_t> parent;
  std::vector<std::uint32_t> col;
  std::vector<float> val;
  std::vector<std::vector<std::int32_t>> rows;
  std::size_t refs() const {
    std::size_t n = 0;
    for (const auto& r : rows) n += r.size();
    return n;
  }
};

// A row's non-zero values as (column, stored value) pairs.
using Sparse = std::vector<std::pair<std::uint32_t, std::uint32_t>>;

template <class T>
std::vector<Sparse> sparse_rows(const std::vector<const T*>& rows, std::size_t len) {
  std::vector<Sparse> out;
  for (const T* row : rows) {
    auto& r = out.emplace_back();
    for (std::size_t j = 0; j < len; ++j) {
      if (row[j] != 0) r.emplace_back(static_cast<std::uint32_t>(j), static_cast<std::uint32_t>(row[j]));
    }
  }
  return out;
}

template <class V>
Lz78 build_lz78(const std::vector<Sparse>& rows, V&& value) {
  Lz78 d;
  std::unordered_map<std::uint64_t, std::int32_t> child;
  const auto key = [](std::int32_t node, std::uint32_t col, std::uint32_t v) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(node + 1)) << 40) | (static_cast<std::uint64_t>(col) << 16) | v;
  };
  for (const Sparse& row : rows) {
    auto& out = d.rows.emplace_back();
    std::size_t j = 0;
    while (j < row.size()) {
      std::int32_t cur = -1;
      std::size_t jj = j;
      while (jj < row.size()) {
        const auto it = child.find(key(cur, row[jj].first, row[jj].second));
        if (it == child.end()) break;
        cur = it->second;
        ++jj;
      }
      if (jj < row.size()) {
        const auto id = static_cast<std::int32_t>(d.parent.size());
        child.emplace(key(cur, row[jj].first, row[jj].second), id);
        d.parent.push_back(cur);
        d.col.push_back(row[jj].first);
        d.val.push_back(value(row[jj].second));
        out.push_back(id);
        j = jj + 1;
      } else {
        out.push_back(cur);
        j = jj;
      }
    }
  }
  return d;
}

// y[col] += sum over the rows of w[r] * row value: weights of the phrases, pushed from each phrase to its parent.
void lz78_left(const Lz78& d, const float* w, std::vector<float>& W, float* y) {
  std::fill(W.begin(), W.end(), 0.f);
  for (std::size_t r = 0; r < d.rows.size(); ++r) {
    if (w[r] == 0.f) continue;
    for (const std::int32_t id : d.rows[r]) W[zs(id)] += w[r];
  }
  for (std::size_t k = d.parent.size(); k-- > 0;) {
    const float x = W[k];
    if (x == 0.f) continue;
    y[d.col[k]] += x * d.val[k];
    if (d.parent[k] >= 0) W[zs(d.parent[k])] += x;
  }
}

// out[r] = sum over the row's columns of value * x[col]: each phrase's partial sum from its parent's.
void lz78_right(const Lz78& d, const float* x, std::vector<float>& S, float* out) {
  for (std::size_t k = 0; k < d.parent.size(); ++k) S[k] = (d.parent[k] >= 0 ? S[zs(d.parent[k])] : 0.f) + d.val[k] * x[d.col[k]];
  for (std::size_t r = 0; r < d.rows.size(); ++r) {
    float s = 0.f;
    for (const std::int32_t id : d.rows[r]) s += S[zs(id)];
    out[r] = s;
  }
}

// RePair over the rows' non-zero (column, value) pairs (pairs never cross rows): terminals are the pairs, rule k replaces the pair
// (a[k], b[k]); each row a sequence of symbols (terminals first, then rules).
struct RePair {
  std::vector<std::uint32_t> tcol;
  std::vector<float> tval;
  std::vector<std::uint32_t> a, b;
  std::vector<std::vector<std::uint32_t>> rows;
  std::size_t terminals() const { return tcol.size(); }
  std::size_t symbols() const {
    std::size_t n = 0;
    for (const auto& r : rows) n += r.size();
    return n;
  }
};

template <class V>
RePair build_repair(const std::vector<Sparse>& rows, V&& value) {
  RePair g;
  std::unordered_map<std::uint64_t, std::uint32_t> term;
  for (const Sparse& row : rows) {
    auto& seq = g.rows.emplace_back();
    seq.reserve(row.size());
    for (const auto& [j, q] : row) {
      const std::uint64_t k = (static_cast<std::uint64_t>(j) << 17) | q;
      auto [it, fresh] = term.try_emplace(k, static_cast<std::uint32_t>(g.tcol.size()));
      if (fresh) {
        g.tcol.push_back(j);
        g.tval.push_back(value(q));
      }
      seq.push_back(it->second);
    }
  }
  // Replace the most frequent pair until none occurs twice (counts of non-overlapping occurrences, left to right).
  std::unordered_map<std::uint64_t, std::uint32_t> count;
  for (;;) {
    count.clear();
    for (const auto& seq : g.rows) {
      for (std::size_t i = 0; i + 1 < seq.size(); ++i) {
        if (i > 0 && seq[i - 1] == seq[i] && seq[i] == seq[i + 1]) {  // aaa: the second pair overlaps the first
          bool odd = true;
          for (std::size_t q = i; q > 0 && seq[q - 1] == seq[i]; --q) odd = !odd;
          if (!odd) continue;
        }
        ++count[(static_cast<std::uint64_t>(seq[i]) << 32) | seq[i + 1]];
      }
    }
    std::uint64_t best = 0;
    std::uint32_t most = 1;
    for (const auto& [k, n] : count) {
      if (n > most || (n == most && n > 1 && k < best)) {
        most = n;
        best = k;
      }
    }
    if (most < 2) break;
    const auto x = static_cast<std::uint32_t>(best >> 32), y = static_cast<std::uint32_t>(best & 0xffffffffu);
    const auto id = static_cast<std::uint32_t>(g.terminals() + g.a.size());
    g.a.push_back(x);
    g.b.push_back(y);
    for (auto& seq : g.rows) {
      std::size_t o = 0;
      for (std::size_t i = 0; i < seq.size();) {
        if (i + 1 < seq.size() && seq[i] == x && seq[i + 1] == y) {
          seq[o++] = id;
          i += 2;
        } else {
          seq[o++] = seq[i++];
        }
      }
      seq.resize(o);
    }
  }
  return g;
}

void repair_left(const RePair& g, const float* w, std::vector<float>& W, float* y) {
  std::fill(W.begin(), W.end(), 0.f);
  for (std::size_t r = 0; r < g.rows.size(); ++r) {
    if (w[r] == 0.f) continue;
    for (const std::uint32_t s : g.rows[r]) W[s] += w[r];
  }
  const std::size_t T = g.terminals();
  for (std::size_t k = g.a.size(); k-- > 0;) {
    const float x = W[T + k];
    if (x == 0.f) continue;
    W[g.a[k]] += x;
    W[g.b[k]] += x;
  }
  for (std::size_t t = 0; t < T; ++t) {
    if (W[t] != 0.f) y[g.tcol[t]] += W[t] * g.tval[t];
  }
}

// The rows as plain sparse lists (CSR, no grammar): y[col] += w * value for every non-zero.
struct Csr {
  std::vector<std::size_t> start;
  std::vector<std::uint32_t> col;
  std::vector<float> val;
};
template <class V>
Csr build_csr(const std::vector<Sparse>& rows, V&& value) {
  Csr c;
  c.start.push_back(0);
  for (const Sparse& r : rows) {
    for (const auto& [j, q] : r) {
      c.col.push_back(j);
      c.val.push_back(value(q));
    }
    c.start.push_back(c.col.size());
  }
  return c;
}
void csr_left(const Csr& c, const float* w, float* y) {
  for (std::size_t r = 0; r + 1 < c.start.size(); ++r) {
    const float x = w[r];
    for (std::size_t i = c.start[r]; i < c.start[r + 1]; ++i) y[c.col[i]] += x * c.val[i];
  }
}
std::size_t csr_bytes(const Csr& c) { return c.col.size() * 3 + c.start.size() * 4; }  // 16-bit column, 8-bit value

// The dense blend of one plane, as the runtime's accumulate_slice (8-bit and fp16), compiled for AVX2 + FMA as the
// runtime's AVX2 build is (the loops vectorise).
[[gnu::target("arch=x86-64-v3"), gnu::noinline]] void dense_u8(const std::uint8_t* q, std::size_t n, float base, float step, float* s) {
  for (std::size_t j = 0; j < n; ++j) s[j] += base + step * static_cast<float>(q[j]);
}
[[gnu::target("arch=x86-64-v3"), gnu::noinline]] void dense_f16(const std::uint16_t* q, std::size_t n, float a, float* s) {
  for (std::size_t j = 0; j < n; ++j) s[j] += a * static_cast<float>(std::bit_cast<std::float16_t>(q[j]));
}
[[gnu::target("arch=x86-64-v3"), gnu::noinline]] void dense_right(const float* W, std::size_t rows, std::size_t cols, const float* x, float* out) {
  for (std::size_t r = 0; r < rows; ++r) {
    float s = 0.f;
    for (std::size_t c = 0; c < cols; ++c) s += W[r * cols + c] * x[c];
    out[r] = s;
  }
}

struct H1Row {
  std::string file, what, form;
  std::size_t dense_bytes = 0, compressed_bytes = 0, phrases = 0, nodes = 0;
  double ms_dense = 0, ms_compressed = 0, max_rel_diff = 0;
};

// Bytes of a compressed form, generously small: an LZ78 node 7 bytes (parent, 16-bit column, 8-bit value), a rule 4
// (two 16-bit symbols when they fit, else 8), a terminal 3, a reference 2 when the dictionary has fewer than 65536
// entries, else 4.
std::size_t lz78_bytes(const Lz78& d) { return d.parent.size() * 7 + d.refs() * (d.parent.size() < 65536 ? 2 : 4); }
std::size_t repair_bytes(const RePair& g) {
  const std::size_t syms = g.terminals() + g.a.size();
  return g.terminals() * 3 + g.a.size() * (syms < 65536 ? 4 : 8) + g.symbols() * (syms < 65536 ? 2 : 4);
}

double rel_diff(const std::vector<float>& a, const std::vector<float>& b) {
  double scale = 1e-12, d = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    scale = std::max(scale, static_cast<double>(std::abs(a[i])));
    d = std::max(d, static_cast<double>(std::abs(a[i] - b[i])));
  }
  return d / scale;
}

// The feature blend of a frame model: per (time slice, channel), the K bases' planes are the rows of one dictionary.
std::vector<H1Row> h1_frame(const fs::path& path, int reps) {
  auto lm = load_model(path);
  if (!lm) throw std::runtime_error(std::format("{}: {}", path.string(), lm.error()));
  const Model& m = *lm;
  const Hyper& h = m.h;
  const int K = h.bases, T = h.grid_t, C = h.feature_channels(), S = h.feature_side();
  const std::size_t plane = zs(S) * zs(S);
  const bool u8 = m.feature_bits == 8;
  const auto plane_index = [&](int k, int t, int c) { return (zs(k) * zs(T) + zs(t)) * zs(C) + zs(c); };
  std::println("{}: {} bases, {} time slices, {} channels, {} x {}, {}-bit", path.filename().string(), K, T, C, S, S, m.feature_bits);
  // Dictionaries per (t, c); the weights of the rows for one frame's blend.
  std::vector<Lz78> lz(zs(T) * zs(C));
  std::vector<RePair> rp(zs(T) * zs(C));
  std::vector<Csr> cs(zs(T) * zs(C));
  std::size_t lz_size = 0, rp_size = 0, lz_nodes = 0, lz_refs = 0, rp_rules = 0, rp_syms = 0, cs_size = 0, cs_nnz = 0;
  for (int t = 0; t < T; ++t) {
    for (int c = 0; c < C; ++c) {
      const std::size_t g = zs(t) * zs(C) + zs(c);
      if (u8) {
        std::vector<const std::uint8_t*> rows;
        for (int k = 0; k < K; ++k) rows.push_back(m.raw_u8.data() + plane_index(k, t, c) * plane);
        const auto v = [](std::uint32_t q) { return static_cast<float>(q); };
        const auto sp = sparse_rows(rows, plane);
        lz[g] = build_lz78(sp, v);
        rp[g] = build_repair(sp, v);
        cs[g] = build_csr(sp, v);
      } else {
        std::vector<const std::uint16_t*> rows;
        for (int k = 0; k < K; ++k) rows.push_back(m.raw_f16.data() + plane_index(k, t, c) * plane);
        const auto v = [](std::uint32_t q) { return static_cast<float>(std::bit_cast<std::float16_t>(static_cast<std::uint16_t>(q))); };
        const auto sp = sparse_rows(rows, plane);
        lz[g] = build_lz78(sp, v);
        rp[g] = build_repair(sp, v);
        cs[g] = build_csr(sp, v);
      }
      lz_size += lz78_bytes(lz[g]);
      rp_size += repair_bytes(rp[g]);
      lz_nodes += lz[g].parent.size();
      lz_refs += lz[g].refs();
      rp_rules += rp[g].a.size();
      rp_syms += rp[g].symbols();
      cs_size += csr_bytes(cs[g]);
      cs_nnz += cs[g].col.size();
    }
  }
  const std::size_t dense_bytes = zs(K) * zs(T) * zs(C) * plane * (u8 ? 1 : 2);
  // One frame: random basis weights, a time between slices 2 and 3 (or 0 and 1).
  std::mt19937_64 rng(5);
  std::uniform_real_distribution<float> U(-1.f, 1.f);
  std::vector<float> wk(zs(K));
  for (float& x : wk) x = U(rng);
  const int i0 = std::min(2, T - 2), i1 = i0 + 1;
  const float ft = 0.37f;
  std::vector<float> yd(zs(C) * plane), yl(yd.size()), yr(yd.size());
  std::vector<float> W;
  const auto dense = [&] {
    std::ranges::fill(yd, 0.f);
    for (int k = 0; k < K; ++k) {
      for (const auto& [t, a] : {std::pair{i0, wk[zs(k)] * (1.f - ft)}, std::pair{i1, wk[zs(k)] * ft}}) {
        for (int c = 0; c < C; ++c) {
          const std::size_t p = plane_index(k, t, c);
          float* s = yd.data() + zs(c) * plane;
          if (u8) {
            const float lo = m.raw_ranges[p * 2], hi = m.raw_ranges[p * 2 + 1];
            dense_u8(m.raw_u8.data() + p * plane, plane, a * lo, a * (hi - lo) / 255.f, s);
          } else {
            dense_f16(m.raw_f16.data() + p * plane, plane, a, s);
          }
        }
      }
    }
  };
  // Compressed: per (t, c), rows weighted by a * step; the constant a * lo of every row added to the plane once.
  std::vector<float> wr(zs(K));
  const auto compressed = [&](int form, std::vector<float>& y) {  // 0 LZ78, 1 RePair, 2 CSR
    std::ranges::fill(y, 0.f);
    for (const auto& [t, f] : {std::pair{i0, 1.f - ft}, std::pair{i1, ft}}) {
      for (int c = 0; c < C; ++c) {
        float base = 0.f;
        for (int k = 0; k < K; ++k) {
          const float a = wk[zs(k)] * f;
          if (u8) {
            const std::size_t p = plane_index(k, t, c);
            const float lo = m.raw_ranges[p * 2], hi = m.raw_ranges[p * 2 + 1];
            wr[zs(k)] = a * (hi - lo) / 255.f;
            base += a * lo;
          } else {
            wr[zs(k)] = a;
          }
        }
        float* s = y.data() + zs(c) * plane;
        if (u8) {
          for (std::size_t j = 0; j < plane; ++j) s[j] += base;
        }
        const std::size_t g = zs(t) * zs(C) + zs(c);
        if (form == 0) {
          W.resize(lz[g].parent.size());
          lz78_left(lz[g], wr.data(), W, s);
        } else if (form == 1) {
          W.resize(rp[g].terminals() + rp[g].a.size());
          repair_left(rp[g], wr.data(), W, s);
        } else {
          csr_left(cs[g], wr.data(), s);
        }
      }
    }
  };
  std::vector<float> yc(yd.size());
  double td = 1e9, tl = 1e9, tr = 1e9, tc = 1e9;
  for (int r = 0; r < reps; ++r) {
    td = std::min(td, timed(dense));
    tl = std::min(tl, timed([&] { compressed(0, yl); }));
    tr = std::min(tr, timed([&] { compressed(1, yr); }));
    tc = std::min(tc, timed([&] { compressed(2, yc); }));
  }
  const std::string name = path.filename().string();
  std::vector<H1Row> out;
  const std::string what = std::format("feature blend, {} bases x 2 time slices x {} channels, {} x {}", K, C, S, S);
  out.push_back({name, what, "LZ78", dense_bytes, lz_size, lz_refs, lz_nodes, 1e3 * td, 1e3 * tl, rel_diff(yd, yl)});
  out.push_back({name, what, "RePair", dense_bytes, rp_size, rp_syms, rp_rules, 1e3 * td, 1e3 * tr, rel_diff(yd, yr)});
  out.push_back({name, what, "CSR", dense_bytes, cs_size, cs_nnz, 0, 1e3 * td, 1e3 * tc, rel_diff(yd, yc)});
  // The first dense layer of the network (grid: C -> H, per pixel W x), as weight tables: right products.
  if (!m.layers.empty()) {
    const Dense& L = m.layers[0];
    std::vector<std::uint16_t> w16(L.w.size());
    for (std::size_t i = 0; i < w16.size(); ++i) w16[i] = std::bit_cast<std::uint16_t>(static_cast<std::float16_t>(L.w[i]));
    // [out][in] rows
    std::vector<const std::uint16_t*> rows;
    for (int o = 0; o < L.out; ++o) rows.push_back(w16.data() + zs(o) * zs(L.in));
    const auto v = [](std::uint32_t q) { return static_cast<float>(std::bit_cast<std::float16_t>(static_cast<std::uint16_t>(q))); };
    const Lz78 d = build_lz78(sparse_rows(rows, zs(L.in)), v);
    std::vector<float> Wf(L.w.size());
    for (std::size_t i = 0; i < Wf.size(); ++i) Wf[i] = v(w16[i]);
    std::vector<float> x(zs(L.in)), od(zs(L.out)), oc(zs(L.out)), Sv(d.parent.size());
    for (float& q : x) q = U(rng);
    constexpr int kPixels = 4096;  // the product per pixel, for a tile of pixels
    double t1 = 1e9, t2 = 1e9;
    for (int r = 0; r < reps; ++r) {
      t1 = std::min(t1, timed([&] {
        for (int p = 0; p < kPixels; ++p) {
          x[zs(p) % x.size()] += 1e-7f;
          dense_right(Wf.data(), zs(L.out), zs(L.in), x.data(), od.data());
        }
      }));
      t2 = std::min(t2, timed([&] {
        for (int p = 0; p < kPixels; ++p) {
          x[zs(p) % x.size()] += 1e-7f;
          lz78_right(d, x.data(), Sv, oc.data());
        }
      }));
    }
    dense_right(Wf.data(), zs(L.out), zs(L.in), x.data(), od.data());
    lz78_right(d, x.data(), Sv, oc.data());
    out.push_back({name, std::format("first layer {} x {}, W x for 4096 pixels", L.out, L.in), "LZ78", L.w.size() * 2, lz78_bytes(d), d.refs(), d.parent.size(),
                   1e3 * t1, 1e3 * t2, rel_diff(od, oc)});
  }
  return out;
}

// A weighted sum of 8-bit rows, dense (AVX2) and from LZ78, RePair and CSR.
std::vector<H1Row> blend_forms(const std::string& name, const std::string& what, const std::vector<const std::uint8_t*>& rows, std::size_t plane,
                               const std::vector<float>& w, int reps) {
  const auto v = [](std::uint32_t x) { return static_cast<float>(x); };
  const auto sp = sparse_rows(rows, plane);
  const Lz78 d = build_lz78(sp, v);
  const RePair g = build_repair(sp, v);
  const Csr c = build_csr(sp, v);
  std::vector<float> yd(plane), yl(plane), yr(plane), yc(plane), W;
  double td = 1e9, tl = 1e9, tr = 1e9, tc = 1e9;
  for (int r = 0; r < reps; ++r) {
    td = std::min(td, timed([&] {
      std::ranges::fill(yd, 0.f);
      for (std::size_t k = 0; k < rows.size(); ++k) dense_u8(rows[k], plane, 0.f, w[k], yd.data());
    }));
    tl = std::min(tl, timed([&] {
      std::ranges::fill(yl, 0.f);
      W.resize(d.parent.size());
      lz78_left(d, w.data(), W, yl.data());
    }));
    tr = std::min(tr, timed([&] {
      std::ranges::fill(yr, 0.f);
      W.resize(g.terminals() + g.a.size());
      repair_left(g, w.data(), W, yr.data());
    }));
    tc = std::min(tc, timed([&] {
      std::ranges::fill(yc, 0.f);
      csr_left(c, w.data(), yc.data());
    }));
  }
  const std::size_t n = rows.size() * plane;
  return {{name, what, "LZ78", n, lz78_bytes(d), d.refs(), d.parent.size(), 1e3 * td, 1e3 * tl, rel_diff(yd, yl)},
          {name, what, "RePair", n, repair_bytes(g), g.symbols(), g.a.size(), 1e3 * td, 1e3 * tr, rel_diff(yd, yr)},
          {name, what, "CSR", n, csr_bytes(c), c.col.size(), 0, 1e3 * td, 1e3 * tc, rel_diff(yd, yc)}};
}

// The stored fine fields of a rollout effect (8-bit, mostly empty): a blend of the start points' fields (rows: starts),
// per field; and the stepper's first convolution as a weight table.
std::vector<H1Row> h1_rollout(const fs::path& path, int reps) {
  const rt::RolloutEffect e = load_effect(path);
  const rollout::Model& m = e.m;
  const int SF = m.h.start_fine;
  std::vector<H1Row> out;
  const std::string name = path.filename().string();
  std::vector<std::vector<std::uint8_t>> q;  // quantised as stored: value / max * 255
  std::vector<float> scale;
  for (const auto& sp : m.starts) {
    for (const auto* f : {&sp.fine_t, &sp.fine_d}) {
      if (f->empty()) continue;
      const float mx = std::max(1e-12f, *std::ranges::max_element(*f));
      auto& r = q.emplace_back();
      for (const float v : *f) r.push_back(static_cast<std::uint8_t>(std::lround(std::clamp(v / mx, 0.f, 1.f) * 255.f)));
      scale.push_back(mx / 255.f);
    }
  }
  if (!q.empty()) {
    const std::size_t plane = zs(SF) * zs(SF);
    std::vector<const std::uint8_t*> rows;
    for (const auto& r : q) rows.push_back(r.data());
    std::mt19937_64 rng(9);
    std::uniform_real_distribution<float> U(0.f, 1.f);
    std::vector<float> w(rows.size());
    for (std::size_t r = 0; r < w.size(); ++r) w[r] = U(rng) * scale[r];
    const std::string what = std::format("blend of {} stored fine fields {} x {}", rows.size(), SF, SF);
    for (auto& r : blend_forms(name, what, rows, plane, w, reps)) out.push_back(std::move(r));
  }
  return out;
}

// Where the crossover lies: 16 synthetic 8-bit fields of 64 x 64, empty except for smooth blobs covering a fraction
// of each (the blobs' shapes repeat across fields, their values do not except where they saturate), blended.
std::vector<H1Row> h1_synthetic(int reps) {
  std::vector<H1Row> out;
  constexpr int kRows = 16, kSide = 64;
  const std::size_t plane = kSide * kSide;
  for (const double cover : {0.5, 0.2, 0.1, 0.05, 0.02}) {
    std::mt19937_64 rng(static_cast<std::uint64_t>(cover * 1000));
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::vector<std::vector<std::uint8_t>> q(kRows, std::vector<std::uint8_t>(plane, 0));
    const double radius = std::sqrt(cover * kSide * kSide / 3.14159265);
    for (int r = 0; r < kRows; ++r) {
      const double cx = 32 + 4 * (U(rng) - 0.5), cy = 32 + 4 * (U(rng) - 0.5), amp = 200 + 100 * U(rng);
      for (int y = 0; y < kSide; ++y) {
        for (int x = 0; x < kSide; ++x) {
          const double d = std::hypot(x - cx, y - cy) / radius;
          if (d < 1.0) q[zs(r)][zs(y) * kSide + zs(x)] = static_cast<std::uint8_t>(std::clamp(amp * (1.0 - d * d), 0.0, 255.0));
        }
      }
    }
    std::vector<const std::uint8_t*> rows;
    for (const auto& r : q) rows.push_back(r.data());
    std::vector<float> w(kRows);
    for (float& x : w) x = static_cast<float>(U(rng));
    for (auto& r : blend_forms("synthetic", std::format("blend of 16 fields 64 x 64, {:.0f}% covered", 100 * cover), rows, plane, w, reps)) out.push_back(std::move(r));
  }
  return out;
}

int h1(const tools::Args& a) {
  const int reps = a.i("reps", 15);
  std::vector<H1Row> rows = h1_synthetic(reps);
  for (const auto& p : a.positional()) {
    std::ifstream in(p, std::ios::binary);
    char magic[8] = {};
    in.read(magic, 8);
    const bool roll = rollout::is_rollout_file(std::span<const char>(magic, 8));
    for (auto& r : roll ? h1_rollout(p, reps) : h1_frame(p, reps)) rows.push_back(std::move(r));
  }
  std::ofstream csv(a.str("out", "results/experiments/h1_compressed_products.csv"));
  csv << "file,product,form,dense_bytes,compressed_bytes,references,dictionary_entries,dense_ms,compressed_ms,max_rel_diff\n";
  std::println("| file | product | form | dense KB | compressed KB | ratio | entries | dense ms | compressed ms | slower by | max rel. diff |");
  std::println("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|");
  for (const H1Row& r : rows) {
    csv << std::format("{},{},{},{},{},{},{},{:.4f},{:.4f},{:.3g}\n", r.file, r.what, r.form, r.dense_bytes, r.compressed_bytes, r.phrases, r.nodes, r.ms_dense,
                       r.ms_compressed, r.max_rel_diff);
    std::println("| {} | {} | {} | {:.1f} | {:.1f} | {:.2f}x | {} | {:.3f} | {:.3f} | {:.1f}x | {:.1e} |", r.file, r.what, r.form, static_cast<double>(r.dense_bytes) / 1024.0,
                 static_cast<double>(r.compressed_bytes) / 1024.0, static_cast<double>(r.dense_bytes) / static_cast<double>(r.compressed_bytes), r.nodes, r.ms_dense,
                 r.ms_compressed, r.ms_compressed / r.ms_dense, r.max_rel_diff);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) try {
  const tools::Args a(argc, argv, {"help", "h1", "h2"});
  if (a.flag("h2")) return h2(a);
  if (a.flag("h1")) return h1(a);
  std::println("nvfx_study_h --h2 [--models DIR] [--reps N] [--frames N] [--isa avx2|avx512|baseline] [--out CSV] | "
               "--h1 [--reps N] [--out CSV] FILES (see the source header)");
  return 0;
} catch (const std::exception& e) {
  std::println(stderr, "nvfx_study_h: {}", e.what());
  return 1;
}
