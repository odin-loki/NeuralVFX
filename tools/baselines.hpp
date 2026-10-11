// The flipbook baselines in the study tools (nvfx_experiment for study A, nvfx_pack for study F, nvfx_f2 for study F2):
// the ladder compared against, which baseline a configuration belongs to, equal-quality comparisons of a method against
// a family of baselines (a flipbook ladder, a codec's quality ladder) with 95% bootstrap intervals over clips, and a
// small parallel loop.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <neuralfx/flipbook.hpp>

#include <atomic>
#include <exception>
#include <format>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <random>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace nfx::tools {

// The ladder: the studies' original 31 configurations (our BC3 layout, raw RGBA8), then the same frame counts and
// resolutions in each production format this build has (BC7, ASTC 4x4 to 12x12; flipbook::ladder_production).
inline std::vector<flipbook::Spec> flipbook_ladder(int size, int frames) {
  auto v = flipbook::ladder(size, frames);
  for (const auto& s : flipbook::ladder_production(size, frames)) v.push_back(s);
  return v;
}

// f(i) for every i in [0, n) on `threads` threads (the caller's among them); the first exception is rethrown once all
// have stopped.
template <class F>
void parallel_for(std::size_t n, int threads, F&& f) {
  std::atomic<std::size_t> next{0};
  std::exception_ptr error;
  std::mutex m;
  const auto work = [&] {
    for (std::size_t i; (i = next++) < n;) {
      try {
        f(i);
      } catch (...) {
        const std::lock_guard lock(m);
        if (!error) error = std::current_exception();
        next = n;
      }
    }
  };
  {
    std::vector<std::jthread> pool;
    for (int t = 1; t < threads; ++t) pool.emplace_back(work);
    work();
  }
  if (error) std::rethrow_exception(error);
}

// One method's points on a set: per clip (bytes, quality) for one or more size measures.
struct Point {
  std::vector<double> q;                         // active PSNR per clip (clip order of the set)
  std::map<std::string, std::vector<double>> b;  // size measure -> bytes per clip
};

// A family of configurations (a flipbook ladder, a codec's quality ladder): config -> point.
using Family = std::map<std::string, Point>;

inline double mean_at(const std::vector<double>& v, const std::vector<std::size_t>& idx) {
  double s = 0;
  for (const std::size_t i : idx) s += v[i];
  return s / static_cast<double>(idx.size());
}

// The best-of-family envelope: best mean quality at or below each mean size, one point per size.
inline std::vector<std::pair<double, double>> envelope(const Family& fam, const std::string& measure, const std::vector<std::size_t>& idx) {
  std::vector<std::pair<double, double>> pts;
  for (const auto& [k, p] : fam) {
    if (!p.b.contains(measure)) continue;
    pts.emplace_back(mean_at(p.b.at(measure), idx), mean_at(p.q, idx));
  }
  std::ranges::sort(pts);
  std::vector<std::pair<double, double>> env;
  for (const auto& [kb, q] : pts) {
    const double best = env.empty() ? q : std::max(env.back().second, q);
    if (!env.empty() && env.back().first == kb) env.back().second = best;
    else env.emplace_back(kb, best);
  }
  return env;
}

// Size along an envelope for a quality: log-linear between points. censor: -1 below the smallest, +1 above the largest.
inline double size_for(const std::vector<std::pair<double, double>>& env, double q, int& censor) {
  censor = 0;
  if (env.empty()) return std::nan("");
  if (env.front().second >= q) {
    censor = -1;
    return env.front().first;
  }
  for (std::size_t i = 1; i < env.size(); ++i) {
    if (env[i].second >= q && env[i - 1].second < q) {
      const double u = (q - env[i - 1].second) / (env[i].second - env[i - 1].second);
      return std::exp(std::log(env[i - 1].first) + u * (std::log(env[i].first) - std::log(env[i - 1].first)));
    }
  }
  censor = 1;
  return env.back().first;
}

// Quality along an envelope at a size (the best at or below it, log-linear between points); NaN below the smallest.
inline double quality_at(const std::vector<std::pair<double, double>>& env, double bytes) {
  if (env.empty() || bytes < env.front().first) return std::nan("");
  for (std::size_t i = 1; i < env.size(); ++i) {
    if (bytes < env[i].first) {
      const double u = (std::log(bytes) - std::log(env[i - 1].first)) / (std::log(env[i].first) - std::log(env[i - 1].first));
      return env[i - 1].second + u * (env[i].second - env[i - 1].second);
    }
  }
  return env.back().second;
}

struct Ratio {
  double point = 0, lo = 0, hi = 0;
  int censor = 0;            // of the point estimate
  double censored_share = 0; // share of resamples outside the envelope
  double other_kb = 0;       // the baseline's size at equal quality (point estimate)
  double dq = 0, dq_lo = 0, dq_hi = 0;  // quality difference at the network's size (network minus baseline envelope)
};

// Equal-quality ratio of a network against a family, with a bootstrap over clips (the same resample on both sides).
// With several families (the video codecs), the best of them: the smallest size at the network's quality, and the
// best quality at its size, each codec along its own ladder.
inline Ratio equal_quality(const Point& net, const std::string& net_measure, const std::vector<const Family*>& fams, const std::string& fam_measure,
                    int resamples = 10000) {
  const std::size_t n = net.q.size();
  std::vector<std::size_t> all(n);
  std::iota(all.begin(), all.end(), 0);
  Ratio r;
  const auto one = [&](const std::vector<std::size_t>& idx, int& censor, double& other, double& dq) {
    const double nb = mean_at(net.b.at(net_measure), idx), nq = mean_at(net.q, idx);
    other = std::numeric_limits<double>::infinity();
    double best_q = -std::numeric_limits<double>::infinity();
    for (const Family* fam : fams) {
      const auto env = envelope(*fam, fam_measure, idx);
      int ce = 0;
      const double kb = size_for(env, nq, ce);
      if (!std::isnan(kb) && kb < other) {
        other = kb;
        censor = ce;
      }
      const double q = quality_at(env, nb);
      if (!std::isnan(q)) best_q = std::max(best_q, q);
    }
    if (std::isinf(other)) other = std::nan("");
    dq = std::isinf(best_q) ? std::nan("") : nq - best_q;
    return other / nb;
  };
  double dq0 = 0;
  r.point = one(all, r.censor, r.other_kb, dq0);
  r.other_kb /= 1024.0;
  r.dq = dq0;
  std::mt19937_64 rng(1);
  std::uniform_int_distribution<std::size_t> pick(0, n - 1);
  std::vector<double> ratios, dqs;
  int cens = 0;
  std::vector<std::size_t> idx(n);
  for (int b = 0; b < resamples; ++b) {
    for (auto& i : idx) i = pick(rng);
    int ce = 0;
    double other = 0, dq = 0;
    const double ratio = one(idx, ce, other, dq);
    if (!std::isnan(ratio)) ratios.push_back(ratio);
    if (!std::isnan(dq)) dqs.push_back(dq);
    cens += ce != 0;
  }
  std::ranges::sort(ratios);
  std::ranges::sort(dqs);
  const auto pct = [](const std::vector<double>& v, double p) {
    return v.empty() ? std::nan("") : v[static_cast<std::size_t>(p * static_cast<double>(v.size() - 1) + 0.5)];
  };
  r.lo = pct(ratios, 0.025);
  r.hi = pct(ratios, 0.975);
  r.dq_lo = pct(dqs, 0.025);
  r.dq_hi = pct(dqs, 0.975);
  r.censored_share = static_cast<double>(cens) / resamples;
  return r;
}

inline Ratio equal_quality(const Point& net, const std::string& net_measure, const Family& fam, const std::string& fam_measure) {
  return equal_quality(net, net_measure, std::vector<const Family*>{&fam}, fam_measure);
}

inline std::string ratio_cell(const Ratio& r) {
  const std::string mark = r.censor > 0 ? ">" : r.censor < 0 ? "<" : "";
  const int prec = r.point < 1 ? 2 : 1;  // the codecs' side: 0.13x reads better than 0.1x
  std::string s = std::format("{}{:.{}f}x [{:.{}f}, {:.{}f}]", mark, r.point, prec, r.lo, prec, r.hi, prec);
  if (r.censored_share > 0.025) s += std::format(" ({:.0f}% censored)", 100 * r.censored_share);
  return s;
}

// The flipbook baselines compared since production encoders were added (docs/REPORT.md §3):
//   bc3_layout  the studies' original ladder: our own BC3-layout encoder and raw RGBA8
//   desktop     adds BC7, the 8-bit-per-pixel format desktop GPUs sample
//   all         adds ASTC at every square block size (mobile GPUs): the strongest flipbook at each size
enum class Baseline { bc3_layout, desktop, all };

inline std::string_view baseline_name(Baseline b) {
  return b == Baseline::bc3_layout ? "BC3 layout and raw (our encoder)" : b == Baseline::desktop ? "with BC7" : "with BC7 and ASTC";
}
inline std::string_view baseline_key(Baseline b) {
  return b == Baseline::bc3_layout ? "bc3layout" : b == Baseline::desktop ? "bc7" : "all";
}

// Whether a flipbook configuration ("bc3 64f 128px", "astc8x8 16f 64px +mv16") belongs to a baseline.
inline bool in_baseline(std::string_view config, Baseline b) {
  const std::string_view codec = config.substr(0, config.find(' '));
  if (codec == "raw" || codec == "bc3") return true;
  if (codec == "bc7") return b != Baseline::bc3_layout;
  return b == Baseline::all;
}

}  // namespace nfx::tools
