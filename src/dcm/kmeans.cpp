// PCA, k-means and labelling comparisons (include/neuralfx/dcm/kmeans.hpp). Ported from the owner's CameraDetector,
// cabinlab/src/diffusion/kmeans.cpp: k-means, its seeding and repairs, the renumbering and the ARI / NMI are the
// original's loops; the PCA is rewritten (covariance plus cyclic Jacobi rotations instead of LibTorch's SVD).
#include <neuralfx/dcm/kmeans.hpp>

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace nfx::dcm {

namespace {

// A row-major view of n x d values.
struct Rows {
  std::span<const double> v;
  std::size_t n = 0, d = 0;
  [[nodiscard]] const double* row(std::size_t i) const { return v.data() + i * d; }
};

// Throws std::invalid_argument for a matrix that is not whole rows, or a non-finite entry: a NaN would make every
// distance comparison false and leave rows with no nearest centroid.
Rows to_rows(std::span<const double> x, std::size_t d, std::string_view what) {
  if (d == 0 || x.size() % d != 0) throw std::invalid_argument(std::format("dcm: {}: needs whole rows of d >= 1 values", what));
  if (!std::ranges::all_of(x, [](double v) { return std::isfinite(v); })) {
    throw std::invalid_argument(std::format("dcm: {}: rows with non-finite values", what));
  }
  return {x, x.size() / d, d};
}

double sq_dist(const double* a, const double* b, std::size_t d) {
  double s = 0.0;
  for (std::size_t j = 0; j < d; ++j) s += (a[j] - b[j]) * (a[j] - b[j]);
  return s;
}

// Uniform in [0, 1) from the top 53 bits: the same on every standard library (std::uniform_real_distribution is not).
double uniform01(std::mt19937_64& rng) { return static_cast<double>(rng() >> 11) * 0x1.0p-53; }

// Nearest of k centroids (row-major k x d; ties: the lower id) and its squared distance.
std::pair<int, double> nearest(const double* x, const std::vector<double>& c, int k, std::size_t d) {
  int best = 0;
  double bd = std::numeric_limits<double>::infinity();
  for (int j = 0; j < k; ++j) {
    const double dist = sq_dist(x, c.data() + static_cast<std::size_t>(j) * d, d);
    if (dist < bd) {
      bd = dist;
      best = j;
    }
  }
  return {best, bd};
}

// Eigen-decomposition of a symmetric d x d matrix (row-major, overwritten) by cyclic Jacobi rotations: on return its
// diagonal holds the eigenvalues and the columns of v the eigenvectors.
void jacobi_eigen(std::vector<double>& a, std::vector<double>& v, std::size_t d) {
  v.assign(d * d, 0.0);
  for (std::size_t i = 0; i < d; ++i) v[i * d + i] = 1.0;
  const auto at = [&](std::size_t r, std::size_t c) -> double& { return a[r * d + c]; };
  for (int sweep = 0; sweep < 100; ++sweep) {
    double off = 0.0, total = 0.0;
    for (std::size_t p = 0; p < d; ++p) {
      for (std::size_t q = 0; q < d; ++q) {
        total += at(p, q) * at(p, q);
        if (p != q) off += at(p, q) * at(p, q);
      }
    }
    if (off <= 1e-30 * total || off == 0.0) break;
    for (std::size_t p = 0; p + 1 < d; ++p) {
      for (std::size_t q = p + 1; q < d; ++q) {
        const double apq = at(p, q);
        if (apq == 0.0) continue;
        // the rotation that zeroes a[p][q] (the smaller of the two angles that do)
        const double theta = (at(q, q) - at(p, p)) / (2.0 * apq);
        const double t = std::abs(theta) > 1e150 ? 0.5 / theta
                                                 : (theta >= 0.0 ? 1.0 : -1.0) / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
        const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
        for (std::size_t k = 0; k < d; ++k) {
          if (k == p || k == q) continue;
          const double akp = at(k, p), akq = at(k, q);
          at(k, p) = at(p, k) = c * akp - s * akq;
          at(k, q) = at(q, k) = s * akp + c * akq;
        }
        at(p, p) -= t * apq;
        at(q, q) += t * apq;
        at(p, q) = at(q, p) = 0.0;
        for (std::size_t k = 0; k < d; ++k) {
          const double vkp = v[k * d + p], vkq = v[k * d + q];
          v[k * d + p] = c * vkp - s * vkq;
          v[k * d + q] = s * vkp + c * vkq;
        }
      }
    }
  }
}

}  // namespace

// --- PCA ------------------------------------------------------------------------------------------------------------

std::vector<double> Pca::project(std::span<const double> x, std::size_t d) const {
  if (d != dims || d == 0 || x.size() % d != 0) {
    throw std::invalid_argument(std::format("dcm: Pca::project: rows of {} values, the fit has {}", d, dims));
  }
  const std::size_t n = x.size() / d, k = count();
  std::vector<double> out(n * k), centred(d);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < d; ++j) centred[j] = x[i * d + j] - mean[j];
    for (std::size_t c = 0; c < k; ++c) {
      const double* comp = components.data() + c * d;
      double s = 0.0;
      for (std::size_t j = 0; j < d; ++j) s += centred[j] * comp[j];
      out[i * k + c] = s;
    }
  }
  return out;
}

Pca fit_pca(std::span<const double> x, std::size_t d, int k) {
  if (k < 1) throw std::invalid_argument("dcm: fit_pca: k must be at least 1");
  if (d == 0 || x.size() % d != 0 || x.size() / d < 2) {
    throw std::invalid_argument("dcm: fit_pca: needs a matrix of two or more rows");
  }
  const auto rows = to_rows(x, d, "fit_pca");
  const std::size_t n = rows.n;
  Pca p;
  p.dims = d;
  p.mean.assign(d, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < d; ++j) p.mean[j] += rows.row(i)[j];
  }
  for (double& m : p.mean) m /= static_cast<double>(n);
  std::vector<double> cov(d * d, 0.0), centred(d);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < d; ++j) centred[j] = rows.row(i)[j] - p.mean[j];
    for (std::size_t a = 0; a < d; ++a) {
      for (std::size_t b = a; b < d; ++b) cov[a * d + b] += centred[a] * centred[b];
    }
  }
  for (std::size_t a = 0; a < d; ++a) {
    for (std::size_t b = a; b < d; ++b) cov[b * d + a] = cov[a * d + b] /= static_cast<double>(n - 1);
  }
  std::vector<double> vec;
  jacobi_eigen(cov, vec, d);
  std::vector<std::size_t> order(d);
  std::iota(order.begin(), order.end(), 0);
  std::ranges::stable_sort(order, [&](std::size_t a, std::size_t b) { return cov[a * d + a] > cov[b * d + b]; });
  const std::size_t keep = std::min({static_cast<std::size_t>(k), n, d});
  p.components.resize(keep * d);
  p.variance.resize(keep);
  for (std::size_t c = 0; c < keep; ++c) {
    const std::size_t col = order[c];
    double* comp = p.components.data() + c * d;
    std::size_t big = 0;  // the first entry of largest magnitude: it is made positive
    for (std::size_t j = 0; j < d; ++j) {
      comp[j] = vec[j * d + col];
      if (std::abs(comp[j]) > std::abs(comp[big])) big = j;
    }
    if (comp[big] < 0.0) {
      for (std::size_t j = 0; j < d; ++j) comp[j] = -comp[j];
    }
    p.variance[c] = std::max(0.0, cov[col * d + col]);
  }
  return p;
}

// --- k-means --------------------------------------------------------------------------------------------------------

namespace {

struct Run {
  std::vector<double> c;  // k x d
  std::vector<int> labels;
  double inertia = 0.0;
  int iterations = 0;
};

// k-means++ seeding: the first centre uniformly, each next one with probability proportional to the squared distance
// to the nearest centre chosen so far (uniformly when every row coincides with a centre).
std::vector<double> seed_centres(const Rows& x, int k, std::mt19937_64& rng) {
  const std::size_t n = x.n, d = x.d;
  std::vector<double> c(static_cast<std::size_t>(k) * d);
  const auto pick_uniform = [&] {
    return std::min(n - 1, static_cast<std::size_t>(uniform01(rng) * static_cast<double>(n)));
  };
  std::size_t pick = pick_uniform();
  std::copy_n(x.row(pick), d, c.begin());
  std::vector<double> d2(n);
  for (std::size_t i = 0; i < n; ++i) d2[i] = sq_dist(x.row(i), c.data(), d);
  for (int j = 1; j < k; ++j) {
    const double total = std::accumulate(d2.begin(), d2.end(), 0.0);
    if (total > 0.0) {
      const double target = uniform01(rng) * total;
      double cum = 0.0;
      pick = n;
      for (std::size_t i = 0; i < n && pick == n; ++i) {
        cum += d2[i];
        if (cum > target) pick = i;
      }
      if (pick == n) {  // rounding left the target past the last step: take the last row not yet a centre
        for (std::size_t i = n; i-- > 0 && pick == n;) {
          if (d2[i] > 0.0) pick = i;
        }
      }
    } else {
      pick = pick_uniform();
    }
    double* cj = c.data() + static_cast<std::size_t>(j) * d;
    std::copy_n(x.row(pick), d, cj);
    for (std::size_t i = 0; i < n; ++i) d2[i] = std::min(d2[i], sq_dist(x.row(i), cj, d));
  }
  return c;
}

Run lloyd(const Rows& x, int k, int max_iter, std::mt19937_64& rng) {
  const std::size_t n = x.n, d = x.d;
  Run r;
  r.c = seed_centres(x, k, rng);
  r.labels.assign(n, -1);
  std::vector<double> dist(n, 0.0);
  const auto assign_all = [&] {  // a row keeps its cluster on a tie, so repaired clusters (duplicates) stay put
    bool changed = false;
    for (std::size_t i = 0; i < n; ++i) {
      auto [j, dd] = nearest(x.row(i), r.c, k, d);
      if (const int cur = r.labels[i];
          cur >= 0 && sq_dist(x.row(i), r.c.data() + static_cast<std::size_t>(cur) * d, d) <= dd) {
        j = cur;
      }
      changed = changed || j != r.labels[i];
      r.labels[i] = j;
      dist[i] = dd;
    }
    return changed;
  };
  assign_all();
  std::vector<std::size_t> count(static_cast<std::size_t>(k));
  while (true) {
    std::ranges::fill(count, 0);
    for (const int l : r.labels) ++count[static_cast<std::size_t>(l)];
    for (int j = 0; j < k; ++j) {
      if (count[static_cast<std::size_t>(j)] > 0) continue;
      // empty cluster: it takes the row farthest from its centroid among clusters that keep a member
      std::size_t far = n;
      for (std::size_t i = 0; i < n; ++i) {
        if (count[static_cast<std::size_t>(r.labels[i])] > 1 && (far == n || dist[i] > dist[far])) far = i;
      }
      --count[static_cast<std::size_t>(r.labels[far])];
      r.labels[far] = j;
      count[static_cast<std::size_t>(j)] = 1;
      dist[far] = 0.0;
    }
    std::ranges::fill(r.c, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      double* cj = r.c.data() + static_cast<std::size_t>(r.labels[i]) * d;
      const double* xi = x.row(i);
      for (std::size_t m = 0; m < d; ++m) cj[m] += xi[m];
    }
    for (int j = 0; j < k; ++j) {
      double* cj = r.c.data() + static_cast<std::size_t>(j) * d;
      for (std::size_t m = 0; m < d; ++m) cj[m] /= static_cast<double>(count[static_cast<std::size_t>(j)]);
    }
    ++r.iterations;
    if (!assign_all() || r.iterations >= max_iter) break;
  }
  r.inertia = std::accumulate(dist.begin(), dist.end(), 0.0);
  return r;
}

}  // namespace

std::vector<int> KMeans::assign(std::span<const double> x, std::size_t d) const {
  if (d != dims) {
    throw std::invalid_argument(std::format("dcm: k-means assign: rows have {} columns, centroids {}", d, dims));
  }
  const auto rows = to_rows(x, d, "k-means assign");
  const int k = clusters();
  std::vector<int> out(rows.n);
  for (std::size_t i = 0; i < rows.n; ++i) out[i] = nearest(rows.row(i), centroids, k, d).first;
  return out;
}

KMeans fit_kmeans(std::span<const double> x, std::size_t d, const KMeansOptions& options) {
  const auto rows = to_rows(x, d, "k-means");
  const int k = options.k;
  if (k < 1 || static_cast<std::size_t>(k) > rows.n) {
    throw std::invalid_argument(std::format("dcm: k-means: k = {} for {} rows", k, rows.n));
  }
  if (options.restarts < 1 || options.max_iter < 1) {
    throw std::invalid_argument("dcm: k-means needs at least one start and one iteration");
  }
  Run best;
  for (int s = 0; s < options.restarts; ++s) {
    std::mt19937_64 rng(options.seed * 1000003ULL + static_cast<std::uint64_t>(s));
    auto r = lloyd(rows, k, options.max_iter, rng);
    if (s == 0 || r.inertia < best.inertia) best = std::move(r);  // the first start is kept even at an infinite inertia
  }
  // number clusters by decreasing size, ties by the first row that belongs to them
  std::vector<std::size_t> size(static_cast<std::size_t>(k), 0), first(static_cast<std::size_t>(k), rows.n);
  for (std::size_t i = 0; i < rows.n; ++i) {
    const auto l = static_cast<std::size_t>(best.labels[i]);
    ++size[l];
    first[l] = std::min(first[l], i);
  }
  std::vector<int> order(static_cast<std::size_t>(k));
  std::iota(order.begin(), order.end(), 0);
  std::ranges::sort(order, [&](int a, int b) {
    const auto ua = static_cast<std::size_t>(a), ub = static_cast<std::size_t>(b);
    return size[ua] != size[ub] ? size[ua] > size[ub] : first[ua] < first[ub];
  });
  std::vector<int> rename(static_cast<std::size_t>(k));
  KMeans out;
  out.dims = rows.d;
  out.centroids.resize(static_cast<std::size_t>(k) * rows.d);
  for (int j = 0; j < k; ++j) {
    const auto old = static_cast<std::size_t>(order[static_cast<std::size_t>(j)]);
    rename[old] = j;
    std::copy_n(best.c.data() + old * rows.d, rows.d, out.centroids.data() + static_cast<std::size_t>(j) * rows.d);
  }
  out.labels.reserve(rows.n);
  for (const int l : best.labels) out.labels.push_back(rename[static_cast<std::size_t>(l)]);
  out.inertia = best.inertia;
  out.iterations = best.iterations;
  return out;
}

// --- comparing labellings --------------------------------------------------------------------------------------------

namespace {

struct Contingency {
  std::map<std::pair<int, int>, double> joint;
  std::map<int, double> a, b;
  double n = 0.0;
};

Contingency contingency(std::span<const int> a, std::span<const int> b, std::string_view what) {
  if (a.size() != b.size()) throw std::invalid_argument(std::format("dcm: {}: labellings differ in size", what));
  Contingency t;
  for (std::size_t i = 0; i < a.size(); ++i) {
    t.joint[{a[i], b[i]}] += 1.0;
    t.a[a[i]] += 1.0;
    t.b[b[i]] += 1.0;
  }
  t.n = static_cast<double>(a.size());
  return t;
}

double pairs(double m) { return m * (m - 1.0) / 2.0; }

}  // namespace

double adjusted_rand_index(std::span<const int> a, std::span<const int> b) {
  const auto t = contingency(a, b, "adjusted_rand_index");
  if (t.n < 2.0) return 1.0;
  double sum_joint = 0.0, sum_a = 0.0, sum_b = 0.0;
  for (const auto& [key, m] : t.joint) sum_joint += pairs(m);
  for (const auto& [key, m] : t.a) sum_a += pairs(m);
  for (const auto& [key, m] : t.b) sum_b += pairs(m);
  const double expected = sum_a * sum_b / pairs(t.n);
  const double max_index = (sum_a + sum_b) / 2.0;
  if (max_index == expected) return 1.0;  // both trivial and identical
  return (sum_joint - expected) / (max_index - expected);
}

double normalized_mutual_info(std::span<const int> a, std::span<const int> b) {
  const auto t = contingency(a, b, "normalized_mutual_info");
  if (t.n == 0.0) return 1.0;
  const auto entropy = [&](const std::map<int, double>& m) {
    double h = 0.0;
    for (const auto& [key, c] : m) h -= c / t.n * std::log(c / t.n);
    return h;
  };
  const double ha = entropy(t.a), hb = entropy(t.b);
  if (ha + hb <= 0.0) return 1.0;  // both a single cluster
  double mi = 0.0;
  for (const auto& [key, c] : t.joint) mi += c / t.n * std::log(c * t.n / (t.a.at(key.first) * t.b.at(key.second)));
  return std::clamp(mi / ((ha + hb) / 2.0), 0.0, 1.0);
}

}  // namespace nfx::dcm
